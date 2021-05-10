#!/usr/bin/env python

from train import *
from histograms import *


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluation')

    parser.add_argument('settings', help='settings file (JSON)')
    parser.add_argument('samples', help='samples file (H5)')

    parser.add_argument('-indices', nargs=2, type=int, default=(0, 1), help='indices range')
    parser.add_argument('-masks', nargs='+', default=['=1', '=2'], help='marginalization masks')

    parser.add_argument('-batch-size', type=int, default=2 ** 12, help='batch size')
    parser.add_argument('-sigma', type=float, default=2e-2, help='relative standard deviation')
    parser.add_argument('-start', type=int, default=2 ** 6, help='start sample')
    parser.add_argument('-stop', type=int, default=2 ** 14, help='end sample')
    parser.add_argument('-groupby', type=int, default=2 ** 8, help='sample group size')

    parser.add_argument('-bins', type=int, default=50, help='number of bins')
    parser.add_argument('-mcmc-limit', type=int, default=int(1e7), help='MCMC size limit')
    parser.add_argument('-wd-limit', type=int, default=int(1e4), help='Wasserstein distance size limit')

    parser.add_argument('-accuracy', default=False, action='store_true')
    parser.add_argument('-coverage', default=False, action='store_true')
    parser.add_argument('-consistence', default=False, action='store_true')
    parser.add_argument('-plots', default=False, action='store_true')
    parser.add_argument('-composition', default=False, action='store_true')

    parser.add_argument('-o', '--output', default=None, help='output file (CSV)')

    args = parser.parse_args()

    # Output
    if args.output is None:
        args.output = args.settings.replace('.json', '_')
        args.output += os.path.basename(args.samples).replace('.h5', '.csv')
    elif os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Simulator & Model
    simulator, model = from_settings(args.settings)

    low, high = simulator.low, simulator.high
    device = low.device

    # Masks
    if type(model) is amsi.NRE:
        masks = torch.tensor([[True] * low.numel()])
    else:
        masks = build_masks(args.masks, low.numel())

    # Samples
    data = amsi.OfflineLTEDataset(args.samples)

    for idx in tqdm(range(*args.indices)):
        theta_star, x_star = data[idx]
        x_star = x_star.to(device)

        index_star = (theta_star - low.cpu()) / (high.cpu() - low.cpu())
        index_star = (args.bins * index_star).long()
        index_star = index_star.clip(max=args.bins - 1)

        ## Ground truth
        if args.accuracy and simulator.tractable:
            pthfile = args.samples.replace('.h5', f'_{idx}.pth')

            if not os.path.exists(pthfile):
                sampler = amsi.TractableSampler(
                    simulator,
                    x_star,
                    batch_size=args.batch_size,
                    sigma=args.sigma * (high - low),
                )

                samples = sampler(args.start, args.stop, groupby=args.groupby)
                truth = reduce_histogramdd(
                    samples, args.bins,
                    low, high,
                    bounded=True,
                    sparse=True,
                    device='cpu',
                ).coalesce()

                torch.save(truth, pthfile)
            else:
                truth = torch.load(pthfile)._coalesced_(True)

            truth, _ = normalize(truth)

            ## Plot
            pdffile = pthfile.replace('.pth', '.pdf')

            if args.plots and not os.path.exists(pdffile):
                corner(
                    pairwise(truth), low.cpu(), high.cpu(),
                    labels=simulator.labels, truth=theta_star,
                    filename=pthfile.replace('.pth', '.pdf'),
                )
        else:
            truth = None

        ## MNRE
        measures = []
        hists = {}
        divergences = {}

        if args.composition:
            pairs = [
                [None] * (i + 1)
                for i in range(theta_star.numel())
            ]
            present = torch.tensor([False] * len(pairs))

        with torch.no_grad():
            model.eval()

            for mask in masks:
                textmask = amsi.mask2str(mask)

                if type(model) is amsi.NRE:
                    nre = model
                else:
                    nre = model[mask]
                    if nre is None:
                        continue

                z_star = model.encoder(x_star)

                ### Hist
                numel = args.bins ** torch.count_nonzero(mask).item()

                sampler = amsi.RESampler(
                    nre,
                    simulator.masked_prior(mask),
                    z_star,
                    batch_size=args.batch_size,
                    sigma=args.sigma * (high[mask] - low[mask]),
                )

                if numel > args.mcmc_limit:
                    samples = sampler(args.start, args.stop, groupby=args.groupby)
                    hist = reduce_histogramdd(
                        samples, args.bins,
                        low, high,
                        bounded=True,
                        sparse=True,
                        device='cpu',
                    ).coalesce()
                else:
                    hist = sampler.histogram(args.bins, low[mask], high[mask])

                ### Quantitative
                hist, total = normalize(hist)

                measures.append({
                    'index': idx,
                    'mask': textmask,
                    'total_probability': total.item(),
                    'entropy': entropy(hist).item(),
                })

                #### Accuracy w.r.t. ground truth
                if truth is not None:
                    dims = torch.arange(len(mask))[~mask]
                    target = marginalize(truth, dim=dims.tolist())

                    if not hist.is_sparse:
                        target = target.to_dense()

                    target = target.to(hist)

                    measures[-1]['entropy_truth'] = entropy(target).item()
                    measures[-1]['kl_truth'] = kl_divergence(target, hist).item()

                    if numel <= args.wd_limit:
                        measures[-1]['wd_truth'] = w_distance(target, hist).item()
                    else:
                        measures[-1]['wd_truth'] = None

                    del target
                else:
                    measures[-1]['entropy_truth'] = None
                    measures[-1]['kl_truth'] = None
                    measures[-1]['wd_truth'] = None

                #### Coverage
                if args.coverage:
                    p = hist[tuple(index_star[mask])]

                    if hist.is_sparse:
                        pdf = hist.values()
                    else:
                        pdf = hist.view(-1)

                    measures[-1]['quantile'] = pdf[pdf >= p].sum().item()
                else:
                    measures[-1]['quantile'] = None

                #### Consistence
                hist = hist.cpu()

                if args.consistence and not hist.is_sparse:
                    divergences[textmask] = {textmask: 0.}

                    for key, (m, h) in hists.items():
                        common = torch.logical_and(mask, m)

                        if torch.all(~common):
                            divergences[textmask][key] = 0.
                            divergences[key][textmask] = 0.
                            continue

                        dims = mask.cumsum(0)[common] - 1
                        p = marginalize(hist, dim=dims.tolist(), keep=True)

                        dims = m.cumsum(0)[common] - 1
                        q = marginalize(h, dim=dims.tolist(), keep=True)

                        divergences[textmask][key] = kl_divergence(p, q).item()
                        divergences[key][textmask] = kl_divergence(q, p).item()

                    hists[textmask] = mask, hist

                ### Qualitative
                if args.plots:
                    labels = [l for (l, m) in zip(simulator.labels, mask) if m]

                    fig = corner(
                        pairwise(hist), low[mask].cpu(), high[mask].cpu(),
                        labels=labels, truth=theta_star[mask],
                        filename=args.output.replace('.csv', f'_{idx}_{textmask}.pdf'),
                    )

                if args.composition and hist.dim() <= 2:
                    present = torch.logical_or(present, mask)
                    indices = torch.nonzero(mask).squeeze()

                    if indices.numel() == 1:
                        i = j = indices.tolist()
                    else:
                        i, j = indices.tolist()

                    pairs[j][i] = hist

        ## Append measures
        df = pd.DataFrame(measures)
        df.to_csv(
            args.output,
            index=False,
            mode='a',
            header=not os.path.exists(args.output),
        )

        ## Export consistence
        if args.consistence:
            df = pd.DataFrame(divergences)
            df.to_csv(args.output.replace('.csv', f'_{idx}.csv'))

        ## Composition
        if args.composition:
            for i in reversed(range(present.numel())):
                if not present[i]:
                    pairs.pop(i)
                    continue

                for j in reversed(range(i + 1)):
                    if not present[j]:
                        pairs[i].pop(j)

            textmask = amsi.mask2str(present)
            labels = [l for (l, m) in zip(simulator.labels, present) if m]

            fig = corner(
                pairs, low[present].cpu(), high[present].cpu(),
                labels=labels, truth=theta_star[present],
                filename=args.output.replace('.csv', f'_{idx}_{textmask}_c.pdf'),
            )

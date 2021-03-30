#!/usr/bin/env python

from train import *
from histograms import *


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluation')

    parser.add_argument('settings', help='settings file (JSON)')
    parser.add_argument('samples', help='samples file (H5)')

    parser.add_argument('-indices', nargs=2, type=int, default=(0, 1), help='indices range')
    parser.add_argument('-masks', nargs='+', default=['=3'], help='marginalization masks')

    parser.add_argument('-batch-size', type=int, default=2 ** 12, help='batch size')
    parser.add_argument('-sigma', type=float, default=0.1, help='sigma')
    parser.add_argument('-start', type=int, default=2 ** 6, help='start sample')
    parser.add_argument('-stop', type=int, default=2 ** 14, help='end sample')
    parser.add_argument('-groupby', type=int, default=2 ** 8, help='sample group size')

    parser.add_argument('-bins', type=int, default=50, help='number of bins')
    parser.add_argument('-limit', type=int, default=int(1e7), help='histogram size limit')

    parser.add_argument('-accuracy', default=False, action='store_true')
    parser.add_argument('-coverage', default=False, action='store_true')
    parser.add_argument('-consistence', default=False, action='store_true')
    parser.add_argument('-plots', default=False, action='store_true')

    parser.add_argument('-o', '--output', default=None, help='output file (CSV)')

    args = parser.parse_args()

    # Output
    if args.output is None:
        args.output = args.settings.replace('.json', '_')
        args.output += os.path.basename(args.samples).replace('.h5', '.csv')
    elif os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Settings
    with open(args.settings) as f:
        settings = json.load(f)

    settings['weights'] = args.settings.replace('.json', '.pth')

    # Simulator & Model
    simulator, model = build_instance(settings)

    low, high = simulator.low, simulator.high
    device = low.device

    # Masks
    masks = build_masks(args.masks, low.numel()).to(device)

    # Samples
    data = amsi.OfflineLTEDataset(args.samples)

    measures = []

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
                    sigma=args.sigma,
                )

                samples = sampler(args.start, args.stop, groupby=args.groupby)
                truth = reduce_histogramdd(
                    samples, args.bins,
                    low, high,
                    bounded=True,
                    sparse=True,
                    device='cpu',
                )

                torch.save(truth, pthfile)
            else:
                truth = torch.load(pthfile).coalesce()

            truth = normalize(truth)

            ## Plot
            pdffile = pthfile.replace('.pth', '.pdf')

            if not os.path.exists(pdffile):
                pairs = get_pairs(truth)

                corner(
                    pairs, low.cpu(), high.cpu(),
                    labels=simulator.labels, truth=theta_star,
                    filename=pthfile.replace('.pth', '.pdf'),
                )

                del pairs
        else:
            truth = None

        ## MNRE
        hists = []
        divergences = []

        with torch.no_grad():
            model.eval()

            for mask in masks:
                size = args.bins ** torch.count_nonzero(mask)
                if size > args.limit:
                    continue

                nre = model[mask]
                if nre is None:
                    continue

                z_star = model.encoder(x_star)

                ### Hist
                sampler = amsi.RESampler(
                    nre,
                    simulator.masked_prior(mask),
                    z_star,
                    batch_size=args.batch_size,
                    sigma=args.sigma,
                )

                hist = sampler.histogram(args.bins, low[mask], high[mask])

                ### Quantitative
                textmask = amsi.mask2str(mask)

                tot = marginalize(hist, dim=[], keep=True)
                hist = normalize(hist)

                measures.append({
                    'index': idx,
                    'mask': textmask,
                    'total_probability': tot.item(),
                    'entropy': entropy(hist).item(),
                })

                #### Accuracy w.r.t. truth
                if truth is not None:
                    dims = torch.arange(len(mask))[mask]
                    target = marginalize(truth, dim=dims.tolist(), keep=True).to_dense().to(hist)

                    measures[-1]['entropy_truth'] = entropy(target).item()
                    measures[-1]['kl_truth'] = kl_divergence(target, hist).item()

                    del target

                #### Coverage
                if args.coverage:
                    i = ravel_multi_index(index_star[mask], hist.shape).item()
                    pdf, order = hist.view(-1).sort(descending=True)
                    order = order.argsort()  # reverse mapping
                    cdf = pdf.cumsum(0)

                    measures[-1]['quantile'] = cdf[order[i]].item()

                #### Consistence
                if args.consistence:
                    divergences.append([])

                    for i, (m, h) in enumerate(hists):
                        h = h.to(hist)
                        common = torch.logical_and(mask, m)

                        if torch.all(~common):
                            divergences[i].append(0.)
                            divergences[-1].append(0.)
                            continue

                        dims = mask.cumsum(0)[common] - 1
                        p = marginalize(hist, dim=dims.tolist(), keep=True)

                        dims = m.cumsum(0)[common] - 1
                        q = marginalize(h, dim=dims.tolist(), keep=True)

                        divergences[i].append(kl_divergence(q, p).item())
                        divergences[-1].append(kl_divergence(p, q).item())

                    divergences[-1].append(0.)

                    hists.append((mask, hist.cpu()))

                ### Qualitative
                if args.plots:
                    pairs = get_pairs(hist)
                    labels = [l for (l, m) in zip(simulator.labels, mask) if m]

                    fig = corner(
                        pairs, low[mask].cpu(), high[mask].cpu(),
                        labels=labels, truth=theta_star[mask],
                        filename=args.output.replace('.csv', f'_{idx}_{textmask}.pdf'),
                    )

                    del pairs

                del hist

        ## Export consistence
        if args.consistence:
            divergences = np.array(divergences)
            np.savetxt(args.output.replace('.csv', f'_{idx}.txt'), divergences, fmt='%.6f')

    # Save
    measures = pd.DataFrame(measures)
    measures.to_csv(args.output, index=False)

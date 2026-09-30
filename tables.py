"""Render bundled summaries or aggregate results from an experiment run."""
import argparse
import csv
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
NAMES = {'shd':'SHD', 'h':'h(W)', 'fit':'Fit', 'f1':'F1', 'time':'Time(s)'}


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def aggregate(table, directory, template):
    """Use the summary layout to aggregate seed-level experiment CSVs."""
    cache = {}
    output = []
    for cell in template:
        method, setting, metric = cell['method'], cell['setting'], cell['metric']
        if table == 1:
            if metric == 'time':
                family, d = 'ER', setting.split('d=')[1]
            else:
                family, d = setting.split(' d=')
            path = directory / ('exp1_raw_{}_d{}.csv'.format(family,d))
            selectors = {'method':method}
            field = {'h':'acyclicity', 'fit':'fit_error', 'time':'time_s'}.get(metric,metric)
        elif table == 2:
            path = directory/'exp2_results.csv'
            selectors = {'config':method.replace(', ', ' + ')}
            field = {'h':'h_w', 'fit':'l_data'}.get(metric,metric)
        elif table == 3:
            path = directory/'exp4_raw.csv'
            selectors = {'method':{'NT':'NOTEARS','DG':'DAGMA','PnP':'DAG-PnP'}[method],
                         'nd_ratio':setting.split('=')[1]}
            field = metric
        else:
            path = directory/'exp4_sachs_results.csv'
            selectors = {'method':method}
            field = {'h':'h_w','fit':'fit_error'}.get(metric,metric)
        if path not in cache:
            cache[path] = read_csv(path)
        rows = [r for r in cache[path] if all(r[k]==v for k,v in selectors.items())]
        if not rows:
            raise ValueError('No results for {} in {}'.format(selectors,path))
        if field not in rows[0]:
            raise ValueError('Missing {} in {}; generate results with rerun.py.'.format(field,path))
        values = [float(r[field]) for r in rows]
        output.append(dict(cell,mean=str(statistics.mean(values)),std=str(statistics.pstdev(values))))
    return output


def format_pair(cell):
    values = []
    for name in ['mean','std']:
        decimals = int(cell[name+'_decimals'])
        unit = Decimal(1).scaleb(-decimals)
        values.append(format(Decimal(cell[name]).quantize(unit,rounding=ROUND_HALF_UP),'f'))
    return ' ± '.join(values)


def layout(table, cells):
    if table == 1:
        headers = ['d','Method'] + [fam+' '+NAMES[m] for fam in ['ER4','SF4']
                   for m in ['shd','h','fit','f1']] + ['Time(s), ER4']
    elif table == 3:
        headers = ['Method'] + ['n/d={} {}'.format(r,NAMES[m]) for r in [1,5,10]
                               for m in ['shd','f1']]
    else:
        headers = ['Configuration' if table==2 else 'Method'] + [NAMES[m] for m in ['shd','h','fit','f1']]
    groups = {}
    for cell in cells:
        key = (cell['setting'].split('d=')[-1],cell['method']) if table==1 else (cell['method'],)
        groups.setdefault(key,[]).append(format_pair(cell))
    return headers, [list(key)+values for key,values in groups.items()]


def main(table):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,help='Aggregate CSVs from a directory produced by rerun.py.')
    parser.add_argument('--format',choices=['markdown','csv','latex'],default='markdown')
    args = parser.parse_args()
    cells = read_csv(ROOT/'results'/('table{}.csv'.format(table)))
    if args.results:
        try:
            cells = aggregate(table,args.results,cells)
        except (OSError,ValueError,KeyError) as exc:
            parser.error(str(exc))
    headers, rows = layout(table,cells)
    if args.format == 'csv':
        import sys
        writer = csv.writer(sys.stdout)
        writer.writerow(headers)
        writer.writerows(rows)
    elif args.format == 'latex':
        def tex(text):
            return text.replace('_',r'\_').replace('±',r'$\pm$')
        print(r'\begin{tabular}{'+'l'*len(headers)+'}')
        print(r'\hline')
        print(' & '.join(map(tex,headers))+r' \\')
        print(r'\hline')
        for row in rows:
            print(' & '.join(map(tex,row))+r' \\')
        print(r'\hline')
        print(r'\end{tabular}')
    else:
        print('| '+' | '.join(headers)+' |')
        print('|'+'|'.join(['---']*len(headers))+'|')
        for row in rows:
            print('| '+' | '.join(row)+' |')

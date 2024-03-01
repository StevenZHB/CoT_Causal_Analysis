# Copyright (c) Guangsheng Bao and Hongbo Zhang.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os.path as path
from mcnemar_test import get_average_treatment_effect, get_accuracy

# generate latex table for cot treatment
def report_cot_treatment_nshot(args):
    nshots = ['cot0shot', 'cot4shot', 'cot8shot', 'cot16shot']
    nshot_names = ['0-Shot', '4-Shot', '8-Shot', '16-Shot']
    datasets = ['Addition:6', 'Product:3']
    datasets = [dataset.replace(':', '_') for dataset in datasets]
    rows = [['Intervention'],
            ['CoT'],
            ['Controlled (w/ default setting)'],
            ['\\quad\\it Treated (w/ golden CoT) $\\uparrow$'],
            ['\\quad\\it Treated (w/ random CoT) $\\downarrow$'],
            ['Controlled (w/ default setting)'],
            ['\\quad\\it Treated (w/ random role)'],
            ['\\quad\\it Treated (w/ random bias)'],
            ['Controlled (w/ golden CoT)'],
            ['\\quad\\it Treated (w/ random role)'],
            ['\\quad\\it Treated (w/ random bias)']]
    for dataset in datasets:
        for nshot in nshots:
            nshot_name = nshot_names[nshots.index(nshot)]
            rows[0].append(f'{nshot_name}')
            # CoT
            acc = get_accuracy(f'{args.outdir}/output.{dataset}.{nshot}.math_teacher.{args.model_name}.json')
            rows[1].append(f'{acc:.3f}')
            # Controlled (w/ default setting): CoT -> Answer?
            control = f'{args.outdir}/output.{dataset}.{nshot}.defaultreason.{args.model_name}.json'
            treat1 = f'{args.outdir}/output.{dataset}.{nshot}.goldreason.{args.model_name}.json'
            treat2 = f'{args.outdir}/output.{dataset}.{nshot}.randomreason.{args.model_name}.json'
            rows[2].append(f'{get_accuracy(control):.3f}')
            if path.exists(treat1):
                base, ate, pvalue = get_average_treatment_effect(control, treat1)
                rows[3].append(f'\\it {ate:+.3f}' + (' *' if pvalue < 0.01 else ''))
            else:
                rows[3].append(f'\\it -')
            base, ate, pvalue = get_average_treatment_effect(control, treat2)
            rows[4].append(f'\\it {ate:+.3f}' + (' *' if pvalue < 0.01 else ''))
            # Controlled (w/ default setting): Instruction -> Answer?
            control = f'{args.outdir}/output.{dataset}.{nshot}.defaultreason.{args.model_name}.json'
            treat1 = f'{args.outdir}/output.{dataset}.{nshot}.defaultreason_randomrole.{args.model_name}.json'
            treat2 = f'{args.outdir}/output.{dataset}.{nshot}.defaultreason_strongbias.{args.model_name}.json'
            rows[5].append(f'{get_accuracy(control):.3f}')
            base, ate, pvalue = get_average_treatment_effect(control, treat1)
            rows[6].append(f'\\it {ate:+.3f}' + (' *' if pvalue < 0.01 else ''))
            base, ate, pvalue = get_average_treatment_effect(control, treat2)
            rows[7].append(f'\\it {ate:+.3f}' + (' *' if pvalue < 0.01 else ''))
            # Controlled (w/ golden CoT): Instruction -> Answer?
            control = f'{args.outdir}/output.{dataset}.{nshot}.goldreason.{args.model_name}.json'
            treat1 = f'{args.outdir}/output.{dataset}.{nshot}.goldreason_randomrole.{args.model_name}.json'
            treat2 = f'{args.outdir}/output.{dataset}.{nshot}.goldreason_strongbias.{args.model_name}.json'
            if path.exists(control):
                rows[8].append(f'{get_accuracy(control):.3f}')
                base, ate, pvalue = get_average_treatment_effect(control, treat1)
                rows[9].append(f'\\it {ate:+.3f}' + (' *' if pvalue < 0.01 else ''))
                base, ate, pvalue = get_average_treatment_effect(control, treat2)
                rows[10].append(f'\\it {ate:+.3f}' + (' *' if pvalue < 0.01 else ''))
            else:
                rows[8].append(f'\\it -')
                rows[9].append(f'\\it -')
                rows[9].append(f'\\it -')
    # format and print
    rows = [' & '.join(row) + ' \\\\' for row in rows]
    for row in rows:
        print(row)


# generate latex table for cot treatment
def report_cot_treatment(args):
    prompt = 'co0shot'
    datasets = ['Addition:6', 'Product:3', 'GSM8K', 'ProofWriter', 'FOLIO', 'LOGIQA']
    datasets = [dataset.replace(':', '_') for dataset in datasets]
    rows = [['Intervention', 'Add.', 'Mult.', 'GSM8K', 'ProofW.', 'FOLIO', 'LOGIQA']]
    # CoT
    accs = [get_accuracy(f'{args.outdir}/output.{dataset}.{prompt}.math_teacher.{args.model_name}.json') for dataset in datasets]
    rows.append(['CoT'] + [f'{acc:.3f}' for acc in accs])
    # Controlled (w/ default setting): CoT -> Answer?
    cols = [['Controlled (w/ default setting)',
             '\\quad\\it Treated (w/ golden CoT) $\\uparrow$',
             '\\quad\\it Treated (w/ random CoT) $\\downarrow$' ]]
    for dataset in datasets:
        control = f'{args.outdir}/output.{dataset}.{prompt}.defaultreason.{args.model_name}.json'
        treat1 = f'{args.outdir}/output.{dataset}.{prompt}.goldreason.{args.model_name}.json'
        treat2 = f'{args.outdir}/output.{dataset}.{prompt}.randomreason.{args.model_name}.json'
        col = [f'{get_accuracy(control):.3f}']
        if path.exists(treat1):
            base, ate, pvalue = get_average_treatment_effect(control, treat1)
            col.append(f'\\it {ate:+.3f}' + (' *' if pvalue < 0.01 else ''))
        else:
            col.append(f'\\it -')
        if path.exists(treat2):
            base, ate, pvalue = get_average_treatment_effect(control, treat2)
            col.append(f'\\it {ate:+.3f}' + (' *' if pvalue < 0.01 else ''))
        else:
            col.append(f'\\it -')
        cols.append(col)
    rows.extend(zip(*cols))
    # Controlled (w/ default setting): Instruction -> Answer?
    cols = [['Controlled (w/ default setting)',
             '\\quad\\it Treated (w/ random role)',
             '\\quad\\it Treated (w/ random bias)']]
    for dataset in datasets:
        control = f'{args.outdir}/output.{dataset}.{prompt}.defaultreason.{args.model_name}.json'
        treat1 = f'{args.outdir}/output.{dataset}.{prompt}.defaultreason_randomrole.{args.model_name}.json'
        treat2 = f'{args.outdir}/output.{dataset}.{prompt}.defaultreason_strongbias.{args.model_name}.json'
        col = [f'{get_accuracy(control):.3f}']
        base, ate, pvalue = get_average_treatment_effect(control, treat1)
        col.append(f'\\it {ate:+.3f}' + (' *' if pvalue < 0.01 else ''))
        base, ate, pvalue = get_average_treatment_effect(control, treat2)
        col.append(f'\\it {ate:+.3f}' + (' *' if pvalue < 0.01 else ''))
        cols.append(col)
    rows.extend(zip(*cols))
    # Controlled (w/ golden CoT): Instruction -> Answer?
    cols = [['Controlled (w/ golden CoT)',
             '\\quad\\it Treated (w/ random role)',
             '\\quad\\it Treated (w/ random bias)']]
    for dataset in datasets:
        control = f'{args.outdir}/output.{dataset}.{prompt}.goldreason.{args.model_name}.json'
        treat1 = f'{args.outdir}/output.{dataset}.{prompt}.goldreason_randomrole.{args.model_name}.json'
        treat2 = f'{args.outdir}/output.{dataset}.{prompt}.goldreason_strongbias.{args.model_name}.json'
        if path.exists(control):
            col = [f'{get_accuracy(control):.3f}']
            base, ate, pvalue = get_average_treatment_effect(control, treat1)
            col.append(f'\\it {ate:+.3f}' + (' *' if pvalue < 0.01 else ''))
            base, ate, pvalue = get_average_treatment_effect(control, treat2)
            col.append(f'\\it {ate:+.3f}' + (' *' if pvalue < 0.01 else ''))
        else:
            col = [f'\\it -'] * 3
        cols.append(col)
    rows.extend(zip(*cols))
    # format and print
    rows = [' & '.join(row) + ' \\\\' for row in rows]
    for row in rows:
        print(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='./exp_cot/output')
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--report', type=str, default='cot_treatment')
    args = parser.parse_args()

    if args.report == 'cot_treatment':
        report_cot_treatment(args)
    elif args.report == 'cot_treatment_nshot':
        report_cot_treatment_nshot(args)

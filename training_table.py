"""Export one run or a paired optimizer comparison to a formatted Excel workbook."""
import argparse
import json
from pathlib import Path

from openpyxl import Workbook
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.table import Table, TableStyleInfo


TRAIN_COLUMNS = (
    ('optimizer', 'Оптимизатор'),
    ('step', 'Шаг'),
    ('tokens_seen', 'Токенов обработано'),
    ('step_tokens', 'Токенов за шаг'),
    ('train_loss', 'Train loss'),
    ('lr', 'AdamW LR'),
    ('muon_lr', 'Muon LR'),
    ('grad_norm_before_clip', 'Норма градиента'),
    ('step_seconds', 'Время шага, с'),
    ('tokens_per_second', 'Токенов/с'),
    ('peak_vram_gb', 'Пик VRAM, ГБ'),
)
MEMORY_COLUMNS = (
    ('step', 'Шаг'),
    ('gpu', 'GPU'),
    ('device_used_gb', 'Используется на GPU, ГБ'),
    ('device_total_gb', 'Всего на GPU, ГБ'),
    ('process_allocated_gb', 'Выделено процессом, ГБ'),
    ('process_reserved_gb', 'Зарезервировано процессом, ГБ'),
    ('step_peak_allocated_gb', 'Пик выделения за шаг, ГБ'),
    ('model_parameters_gb', 'Веса модели, ГБ'),
    ('gradients_gb', 'Градиенты, ГБ'),
    ('optimizer_states_gb', 'Состояния оптимизатора, ГБ'),
    ('model_buffers_gb', 'Буферы модели, ГБ'),
    ('other_current_allocated_gb', 'Прочая текущая память, ГБ'),
    ('other_at_step_peak_approx_gb', 'Прочая память на пике, оценка, ГБ'),
)
WIDTHS = {
    'Оптимизатор': 14, 'Шаг': 10, 'Токенов обработано': 22, 'Токенов за шаг': 18,
    'Train loss': 14, 'AdamW LR': 16, 'Muon LR': 16, 'Норма градиента': 19,
    'Время шага, с': 16, 'Токенов/с': 16, 'Пик VRAM, ГБ': 16,
    'GPU': 9, 'Используется на GPU, ГБ': 23, 'Всего на GPU, ГБ': 18,
    'Выделено процессом, ГБ': 24, 'Зарезервировано процессом, ГБ': 29,
    'Пик выделения за шаг, ГБ': 25, 'Веса модели, ГБ': 18,
    'Градиенты, ГБ': 17, 'Состояния оптимизатора, ГБ': 25, 'Буферы модели, ГБ': 20,
    'Прочая текущая память, ГБ': 26,
    'Прочая память на пике, оценка, ГБ': 34,
    'Первый: train loss': 20, 'Второй: train loss': 20, 'Разница train loss': 20,
    'Первый: val loss': 20, 'Второй: val loss': 20, 'Разница val loss': 20,
    'Первый: время шага, с': 23, 'Второй: время шага, с': 24,
}


def style_sheet(sheet, title, columns, table_name):
    sheet.title = title
    for index, (_, label) in enumerate(columns, start=1):
        sheet.cell(row=1, column=index, value=label)
    sheet.freeze_panes = 'A2'
    sheet.sheet_view.showGridLines = False
    for cell in sheet[1]:
        cell.fill = PatternFill('solid', fgColor='17365D')
        cell.font = Font(color='FFFFFF', bold=True)
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
    sheet.row_dimensions[1].height = 34
    for index, (_, label) in enumerate(columns, start=1):
        sheet.column_dimensions[sheet.cell(1, index).column_letter].width = WIDTHS.get(label, 18)
    if sheet.max_row > 1:
        table = Table(displayName=table_name, ref=f'A1:{sheet.cell(sheet.max_row, sheet.max_column).coordinate}')
        table.tableStyleInfo = TableStyleInfo(name='TableStyleMedium2', showFirstColumn=False,
                                               showLastColumn=False, showRowStripes=True,
                                               showColumnStripes=False)
        sheet.add_table(table)
        sheet.auto_filter.ref = table.ref
    sheet.sheet_properties.pageSetUpPr.fitToPage = True
    sheet.page_setup.fitToWidth = 1
    sheet.page_setup.fitToHeight = 0


def load_events(log_path):
    with log_path.open('r', encoding='utf-8') as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f'Invalid JSON on line {line_number} in {log_path}: {error}') from error


def optimizer_label(events, fallback):
    for event in events:
        if event.get('event') == 'start' and event.get('optimizer'):
            name = event['optimizer'].lower()
            return 'AdamW' if name == 'adamw' else 'Muon' if name == 'muon' else name.title()
    return fallback


def add_run_sheets(workbook, label, events, suffix):
    training_title = f'{label} обучение'[:31]
    memory_title = f'{label} GPU'[:31]
    training = workbook.create_sheet(training_title)
    style_sheet(training, training_title, TRAIN_COLUMNS, f'Training{suffix}')
    memory = workbook.create_sheet(memory_title)
    style_sheet(memory, memory_title, MEMORY_COLUMNS, f'Memory{suffix}')

    for record in events:
        if record.get('event') != 'train':
            continue
        training.append([label if key == 'optimizer' else record.get(key)
                         for key, _ in TRAIN_COLUMNS])
        for gpu_stats in record.get('gpu_memory', []):
            values = []
            for key, _ in MEMORY_COLUMNS:
                if key == 'step':
                    values.append(record.get('step'))
                elif key == 'optimizer_states_gb':
                    values.append(gpu_stats.get(key, gpu_stats.get('adamw_states_gb')))
                else:
                    values.append(gpu_stats.get(key))
            memory.append(values)

    style_sheet(training, training_title, TRAIN_COLUMNS, f'Training{suffix}')
    style_sheet(memory, memory_title, MEMORY_COLUMNS, f'Memory{suffix}')
    train_formats = {
        'Train loss': '0.0000', 'AdamW LR': '0.00000000', 'Muon LR': '0.00000000',
        'Норма градиента': '0.000', 'Время шага, с': '0.00',
        'Токенов/с': '#,##0', 'Пик VRAM, ГБ': '0.000',
        'Токенов обработано': '#,##0', 'Токенов за шаг': '#,##0',
    }
    memory_formats = {label: '0.000' for _, label in MEMORY_COLUMNS if label != 'GPU'}
    for sheet, formats in ((training, train_formats), (memory, memory_formats)):
        for column_index, header in enumerate(sheet[1], start=1):
            number_format = formats.get(header.value)
            if number_format:
                for row in sheet.iter_rows(min_row=2, min_col=column_index, max_col=column_index):
                    row[0].number_format = number_format
    if training.max_row > 1:
        loss_col = next(index for index, (_, col_label) in enumerate(TRAIN_COLUMNS, start=1)
                        if col_label == 'Train loss')
        training.conditional_formatting.add(
            f'{training.cell(2, loss_col).coordinate}:{training.cell(training.max_row, loss_col).coordinate}',
            ColorScaleRule(start_type='min', start_color='C6EFD6',
                           mid_type='percentile', mid_value=50, mid_color='FFF2CC',
                           end_type='max', end_color='F4CCCC'))
    return label, events


def add_comparison_sheet(workbook, first, second, first_name, second_name):
    first_train = {event['tokens_seen']: event for event in first if event.get('event') == 'train'}
    second_train = {event['tokens_seen']: event for event in second if event.get('event') == 'train'}
    first_val = {event['tokens_seen']: event for event in first if event.get('event') == 'validation'}
    second_val = {event['tokens_seen']: event for event in second if event.get('event') == 'validation'}
    tokens = sorted(set(first_train) | set(second_train) | set(first_val) | set(second_val))

    columns = (
        ('tokens_seen', 'Токенов обработано'),
        ('first_train_loss', f'{first_name}: train loss'),
        ('second_train_loss', f'{second_name}: train loss'),
        ('train_loss_delta', 'Разница train loss'),
        ('first_val_loss', f'{first_name}: val loss'),
        ('second_val_loss', f'{second_name}: val loss'),
        ('val_loss_delta', 'Разница val loss'),
        ('first_step_seconds', f'{first_name}: время шага, с'),
        ('second_step_seconds', f'{second_name}: время шага, с'),
    )
    sheet = workbook.create_sheet('Сравнение')
    style_sheet(sheet, 'Сравнение', columns, 'OptimizerComparison')
    for token_count in tokens:
        a_train, b_train = first_train.get(token_count), second_train.get(token_count)
        a_val, b_val = first_val.get(token_count), second_val.get(token_count)
        a_loss = a_train.get('train_loss') if a_train else None
        b_loss = b_train.get('train_loss') if b_train else None
        a_val_loss = a_val.get('val_loss') if a_val else None
        b_val_loss = b_val.get('val_loss') if b_val else None
        sheet.append([
            token_count, a_loss, b_loss,
            b_loss - a_loss if a_loss is not None and b_loss is not None else None,
            a_val_loss, b_val_loss,
            b_val_loss - a_val_loss if a_val_loss is not None and b_val_loss is not None else None,
            a_train.get('step_seconds') if a_train else None,
            b_train.get('step_seconds') if b_train else None,
        ])
    style_sheet(sheet, 'Сравнение', columns, 'OptimizerComparison')
    for row in sheet.iter_rows(min_row=2):
        row[0].number_format = '#,##0'
        for cell in row[1:7]:
            cell.number_format = '0.0000'
        row[7].number_format = '0.00'
        row[8].number_format = '0.00'
    for label in ('Разница train loss', 'Разница val loss'):
        column = next(index for index, (_, col_label) in enumerate(columns, start=1)
                      if col_label == label)
        if sheet.max_row > 1:
            sheet.conditional_formatting.add(
                f'{sheet.cell(2, column).coordinate}:{sheet.cell(sheet.max_row, column).coordinate}',
                ColorScaleRule(start_type='min', start_color='C6EFD6',
                               mid_type='num', mid_value=0, mid_color='FFF2CC',
                               end_type='max', end_color='F4CCCC'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logfile', type=Path, help='First run metrics.jsonl (AdamW for comparison)')
    parser.add_argument('--compare', type=Path, help='Second run metrics.jsonl (Muon for comparison)')
    parser.add_argument('--output', type=Path,
                        help='Output .xlsx path (defaults beside the first log)')
    args = parser.parse_args()
    if not args.logfile.is_file():
        parser.error(f'log file not found: {args.logfile}')
    if args.compare is not None and not args.compare.is_file():
        parser.error(f'comparison log not found: {args.compare}')

    output_name = 'optimizer_comparison.xlsx' if args.compare else 'training_metrics.xlsx'
    output = args.output or args.logfile.with_name(output_name)
    if output.suffix.lower() != '.xlsx':
        parser.error('output filename must end with .xlsx')
    output.parent.mkdir(parents=True, exist_ok=True)

    first_events = list(load_events(args.logfile))
    second_events = list(load_events(args.compare)) if args.compare else None
    first_name = optimizer_label(first_events, 'Первый')
    second_name = optimizer_label(second_events, 'Второй') if second_events else None
    workbook = Workbook()
    workbook.remove(workbook.active)
    add_run_sheets(workbook, first_name, first_events, 'First')
    if second_events is not None:
        add_run_sheets(workbook, second_name, second_events, 'Second')
        add_comparison_sheet(workbook, first_events, second_events, first_name, second_name)
    workbook.properties.title = 'Сравнение оптимизаторов' if second_events else 'Результаты обучения модели'
    workbook.properties.subject = 'Метрики шагов и память GPU'
    workbook.save(output)
    first_steps = sum(event.get('event') == 'train' for event in first_events)
    if second_events is not None:
        second_steps = sum(event.get('event') == 'train' for event in second_events)
        print(f'Excel-файл сохранён: {output} ({first_name}: {first_steps} шагов, '
              f'{second_name}: {second_steps} шагов)')
    else:
        print(f'Excel-файл сохранён: {output} ({first_steps} шагов)')


if __name__ == '__main__':
    main()

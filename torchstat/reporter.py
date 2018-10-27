import pandas as pd


pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 10000)

def flops_to_string(flops):
    if flops // 10**9 > 0:
        return str(round(flops / 10.**9, 2)) + 'GFlops'
    elif flops // 10**6 > 0:
        return str(round(flops / 10.**6, 2)) + 'MFlops'
    elif flops // 10**3 > 0:
        return str(round(flops / 10.**3, 2)) + 'KFlops'
    return str(flops) + 'Flops'


def report_format(collected_nodes):
    data = list()
    for node in collected_nodes:
        name = node.name
        input_shape = ' '.join(['{:>3d}'] * len(node.input_shape)).format(
            *[e for e in node.input_shape])
        output_shape = ' '.join(['{:>3d}'] * len(node.output_shape)).format(
            *[e for e in node.output_shape])
        parameter_quantity = node.parameter_quantity
        inference_memory = node.inference_memory
        MAdd = node.MAdd
        Flops = node.Flops
        duration = node.duration
        data.append([name, input_shape, output_shape,
                     parameter_quantity, inference_memory, MAdd, duration, Flops])
    df = pd.DataFrame(data)
    df.columns = ['module name', 'input shape', 'output shape',
                  'parameter quantity', 'inference memory(MB)',
                  'MAdd', 'duration', 'Flops']
    df['duration percent'] = df['duration'] / (df['duration'].sum() + 1e-7)
    total_parameters_quantity = df['parameter quantity'].sum()
    total_memory = df['inference memory(MB)'].sum()
    total_operation_quantity = df['MAdd'].sum()
    total_flops = df['Flops'].sum()
    total_duration = df['duration percent'].sum()
    del df['duration']

    # Add Total row
    total_df = pd.Series([total_parameters_quantity, total_memory, total_operation_quantity, total_flops, total_duration],
                         index=['parameter quantity', 'inference memory(MB)', 'MAdd', 'Flops', 'duration percent'],
                         name='total')
    df = df.append(total_df)

    df = df.fillna(' ')
    df['inference memory(MB)'] = df['inference memory(MB)'].apply(
        lambda x: '{:.2f}'.format(x))
    df['duration percent'] = df['duration percent'].apply(lambda x: '{:.2%}'.format(x))
    df['MAdd'] = df['MAdd'].apply(lambda x: '{:,}'.format(x))
    df['Flops'] = df['Flops'].apply(lambda x: '{:,}'.format(x))

    summary = str(df) + '\n'
    summary += "=" * len(str(df).split('\n')[0])
    summary += '\n'
    summary += "Total params: {:,}\n".format(total_parameters_quantity)

    summary += "-" * len(str(df).split('\n')[0])
    summary += '\n'
    summary += "Total memory: {:.2f}MB\n".format(total_memory)
    summary += "Total MAdd: {:,}\n".format(total_operation_quantity)
#    summary += "Total Flops: {:,}\n".format(total_flops)
    summary += "Total Flops: {}\n".format(flops_to_string(total_flops))
    return summary

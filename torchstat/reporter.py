import pandas as pd


pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 10000)


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
        duration = node.duration
        data.append([name, input_shape, output_shape,
                     parameter_quantity, inference_memory, MAdd, duration])
    df = pd.DataFrame(data)
    df.columns = ['module name', 'input shape', 'output shape',
                  'parameter quantity', 'inference memory(MB)',
                  'MAdd', 'duration']
    df['duration percent'] = df['duration'] / (df['duration'].sum() + 1e-7)
    total_parameters_quantity = df['parameter quantity'].sum()
    total_memory = df['inference memory(MB)'].sum()
    total_operation_quantity = df['MAdd'].sum()
    del df['duration']
    df = df.fillna(' ')
    df['inference memory(MB)'] = df['inference memory(MB)'].apply(
        lambda x: '{:.2f}MB'.format(x))
    df['duration percent'] = df['duration percent'].apply(lambda x: '{:.2%}'.format(x))
    df['MAdd'] = df['MAdd'].apply(lambda x: '{:,}'.format(x))

    summary = str(df) + '\n'
    summary += "=" * len(str(df).split('\n')[0])
    summary += '\n'
    summary += "total parameters quantity: {:,}\n".format(total_parameters_quantity)
    summary += "total memory: {:.2f}MB\n".format(total_memory)
    summary += "total MAdd: {:,}\n".format(total_operation_quantity)
    return summary



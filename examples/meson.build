examples = {
    'cusbf-main': {
        'src': 'main.cu',
    },
}

foreach name, config : examples
    executable(
        name,
        config['src'],
        dependencies: [cusbf_dep, cli11_dep] + config.get('deps', []),
        cuda_args: cuda_args,
        install: true,
    )
endforeach

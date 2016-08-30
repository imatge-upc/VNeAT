import yaml
from os.path import join

CONFIG_PATH = join("..", "..", "config", "adContinuum.yaml")

with open(CONFIG_PATH, 'r') as conf_file:
    try:
        conf = yaml.load(conf_file)

        # INPUT
        print '\nINPUT'
        input_paths = conf['input']
        for key, value in input_paths.iteritems():
            print key, ": ", value

        # MODEL
        print '\nMODEL'
        model = conf['model']
        for key, value in model.iteritems():
            print key, ": ", value

        # PROCESSING PARAMS
        print '\nPROCESSING PARAMETERS'
        processing_params = conf['processing_params']
        for key, value in processing_params.iteritems():
            print key, ": ", value

        # OUTPUT
        print '\nOUTPUT'
        output = conf['output']
        for key, value in output.iteritems():
            print key, ": ", value

    except yaml.YAMLError as exc:
        print(exc)


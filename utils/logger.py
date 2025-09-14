import logging
import yaml
import os

def load_config():
    try: 
        with open('config.yaml','r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print("Warning: config.yaml not found. Using default logging settings.")
        
    except Exception as e:
        print(f"Error loading config : {e}")
        return {
            'logging':{
                'level':'INFO',
                'file':'logs/app.log'
            }
        }

#load config from yaml file 
config=load_config()

#setting up log dir
log_file=config['logging'].get('file','logs/app.log')
os.makedirs(os.path.dirname(log_file),exist_ok=True)

# configure logging
logging.basicConfig(
    filename=log_file,
    level=getattr(logging,config['logging'].get('level','INFO')),
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger=logging.getLogger(__name__)

# for console
console_handler = logging.StreamHandler()
console_handler.setLevel(getattr(logging,config['logging'].get('level','INFO')))
formatter=logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
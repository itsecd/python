
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
env_path = Path('.')/'.env'
load_dotenv(dotenv_path=env_path)


FOLDER_TIGER = os.getenv("FOLDER_TIGER")

FOLDER_LEOPARD = os.getenv("FOLDER_LEOPARD")

COUNT_FOR_LIST=os.getenv("COUNT_FOR_LIST")
COUNT_FOR_DOWNLOADS=os.getenv("COUNT_FOR_DOWNLOADS")

SEARCH_TIGER=os.getenv("SEARCH_TIGER")
SEARCH_LEOPARD=os.getenv("SEARCH_LEOPARD")


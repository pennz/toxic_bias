import subprocess
import os
import gc
import psutil

USER_NAME='pengyu'
TFRECORD_FILDATALAG = '.tf_record_saved'

def run_commans(commands, timeout=30):
    for c in commands.splitlines():
        c = c.strip()
        if c.startswith("#"):
            continue
        stdout, stderr = run_process(c,timeout)
        if stdout:
            print(stdout.decode('utf-8'))
        if stderr:
            print(stderr.decode('utf-8'))
            print("stop at command {}, as it reports error in stderr".format(c))
            break

def run_process(process_str,timeout):
    print("{}:{}$ ".format(USER_NAME, os.getcwd())+process_str)
    MyOut = subprocess.Popen(process_str,
                             shell=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
    try:
        MyOut.wait(timeout=timeout)
        return MyOut.communicate()
    except subprocess.TimeoutExpired as e:
        return None, bytes('\'{}\''.format(e), 'utf-8')

def run_process_print(process_str,timeout=30):
    stdout, stderr = run_process(process_str, timeout)
    if stdout:
        print(stdout.decode('utf-8'))
    if stderr:
        print(stderr.decode('utf-8'))

def pip_install_thing():
    to_installs = """pip install --upgrade pip
    #pip install tensorflow-gpu==2.0.0-alpha0
    #pip install drive-cli"""
    run_commans(to_installs, timeout=60*10)

#pip_install_thing()

def upload_file_one_at_a_time(file_name, saved_name=None):
    if not saved_name:
        saved_name = file_name.split('/')[-1]
    run_process_print("curl  --header \"File-Name:{1}\" --data-binary @{0} http://97.64.108.66:8001".format(file_name, saved_name))
    #print("You need to goto VPS and change filename")

def download_file_one_at_a_time(file_name, directory=".", overwrite=False):
    if overwrite:
        run_process_print("wget http://97.64.108.66:8000/{0} -O \"{1}/{0}\"".format(file_name, directory))
    else:
        run_process_print("[ -f {1}/{0} ] || wget http://97.64.108.66:8000/{0} -P {1}".format(file_name, directory))

#upload_file_one_at_a_time("env_prepare.py")

def setup_kaggle():
    s = """pip install kaggle 
    mkdir $HOME/.kaggle 
    echo '{"username":"k1gaggle","key":"f51513f40920d492e9d81bc670b25fa9"}' > $HOME/.kaggle/kaggle.json
    chmod 600 $HOME/.kaggle/kaggle.json
    """
    run_commans(s, timeout=60)

def setup_gdrive():
    download_file_one_at_a_time("gdrive")
    s = """chmod +x ./gdrive
    mkdir $HOME/.gdrive 
    chmod +x ./gdrive
    """
    run_commans(s)
    #download_file_one_at_a_time("token_v2.json", "$HOME/.gdrive")
    str= """{
        "access_token": "ya29.GlsWB6DpEzK1qbegW-7FGy84GUtdR8O57aoq3i73DiFLlwpGxG1hZGwCVLiBIFNCDIw0zgQ6Fs4aBkf1YWbc30_yJMLCtv1E1b20nqMF2gRF3cJU_Ks-xnsaF5WV",
        "token_type": "Bearer",
        "refresh_token": "1/uxgj61NZOFM_LkIZd6QHpGX0Nj8bm9004DK68Ywu0pU",
        "expiry": "2019-05-27T06:11:29.604819094-04:00"
    }"""
    with open("token_v2.json", 'wb') as f:
            f.write(bytes(str, 'utf-8'))
    run_process_print('mv token_v2.json $HOME/.gdrive') # last command cannot know $HOME easily, so python + shell

#setup_gdrive()
#upload_file_one_at_a_time("~/.kaggle/kaggle.json")
#upload_file_one_at_a_time("env_prepare.py")

#upload_file_one_at_a_time("/sync/AI/dog-breed/kaggle-dog-breed/src/solver/server.py")
#download_file_one_at_a_time("kaggle.json")
#download_file_one_at_a_time("server.py")

def do_gc():
    process = psutil.Process(os.getpid())
    print('start', process.memory_info().rss)
    gc.collect()
    print('called gc collect', process.memory_info().rss)

def list_submisstion():
    run_process_print("kaggle competitions submissions -c jigsaw-unintended-bias-in-toxicity-classification")


def get_mem_hogs():
    '''get memory usage by ipython objects

    :return: sorted ipython objects (name, mem_usage) list
    '''
    import sys
    # These are the usual ipython objects, including this one you are creating
    ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']

    # Get a sorted list of the objects and their sizes
    return sorted([(x, sys.getsizeof(globals().get(x))) for x in dir() if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)

def download_lstm_from_gdrive():
    """
    download from gdrive, files are in `lstm_data` folder
    """
    run_commans(
        """
        ./gdrive download  -r 1KzpZTVJunVCSdOFvHZrA4O5nIeDp0gsl
        ./gdrive download 1qOKKUaAanTegKyRqn5llE871pYxkCYOj  # predicts result (for target)
        ./gdrive download 1x4sqy4nxX5l-r1nWlV-qU4PGn7J4rz59  # lstm model, to predict target
        ./gdrive download 1Qg_xU3CWJx1670fF9qyUlETqB59nSgEu  # identity model
        mv lstm_data/* . && touch """+TFRECORD_FILDATALAG
        ,
        timeout=60*10
    )

def update_src():
    run_process_print('rm -rf __pycache__ /proc/driver/nvidia/identity-model/events*')
    download_file_one_at_a_time("data_prepare.py", overwrite=True)
    download_file_one_at_a_time("lstm.py", overwrite=True)
    download_file_one_at_a_time("env_prepare.py", overwrite=True)

quick = False
if os.getcwd().find('lstm') > 0:
    #upload_file_one_at_a_time("data_prepare.py")
    setup_gdrive()
else:
    #do_gc()
    if not os.path.isfile('.env_setup_done'):
        if not quick:
            setup_kaggle()
            list_submisstion()
            pip_install_thing()
            #download_lstm_from_gdrive()
        update_src()
        setup_gdrive()
        run_process_print('export PATH=$PWD:$PATH') # not helpful
        run_process_print('touch .env_setup_done')


#get_ipython().reset()  # run in ipython

import subprocess
import os
import gc
import psutil

USER_NAME='pengyu'
TFRECORD_FILDATALAG = '.tf_record_saved'
GDRIVE_DOWNLOAD_DEST = '/proc/driver/nvidia'

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
    #pip install -q tensorflow-gpu==2.0.0-alpha0
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

def download_kaggle_data():
    if not os.path.isfile('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv'):
        s = """kaggle competitions download jigsaw-unintended-bias-in-toxicity-classification
        unzip test.csv.zip
        unzip train.csv.zip
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

def mount_gdrive():
    from google.colab import drive
    drive.mount('/content/gdrivedata')

    run_process_print(f'touch {TFRECORD_FILDATALAG}')


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
        f"""
        ./gdrive download 1glTXC4_DCE3DGJaGT721CcPAkE9RTkOi --path {GDRIVE_DOWNLOAD_DEST} # train
        ./gdrive download 1mMwuBOLNqa_gaY2O7v3jpOk-DFgna-1E --path {GDRIVE_DOWNLOAD_DEST} # test
        ./gdrive download 1WAyOTiG3rvsrp1MDeacwikoXTH7XqEtQ --path {GDRIVE_DOWNLOAD_DEST} # embedding
        ./gdrive download 1d_2uUzStUhuzErWAcIIk2TuzA1bFyKN7 --path {GDRIVE_DOWNLOAD_DEST} # predicts (no res)
        ./gdrive download 1VFYcLECsE2BAYoe_q2o4a7aMT3OTp5S6 --path {GDRIVE_DOWNLOAD_DEST} # model
        #./gdrive download   # predicts result (for target)
        #./gdrive download   # identity model
        #mv lstm_data/* . 
        touch """+TFRECORD_FILDATALAG
        ,
        timeout=60*10
    )

def up():
    run_process_print('rm -rf __pycache__ /proc/driver/nvidia/identity-model/events*')
    download_file_one_at_a_time("data_prepare.py", overwrite=True)
    download_file_one_at_a_time("lstm.py", overwrite=True)
    download_file_one_at_a_time("env_prepare.py", overwrite=True)

def exit00():
    import os
    os._exit(00)  # will make ipykernel restart

quick = False
if os.getcwd().find('lstm') > 0:
    #upload_file_one_at_a_time("data_prepare.py")
    setup_gdrive()
else:
    #do_gc()
    if not os.path.isfile('.env_setup_done'):
        if not quick:
            setup_kaggle()
            if not os.path.isdir("../input"):
                download_kaggle_data()
            list_submisstion()
            pip_install_thing()
            try:
                mount_gdrive()
            except ModuleNotFoundError:
                setup_gdrive()
                download_lstm_from_gdrive()
        run_process_print('export PATH=$PWD:$PATH') # not helpful
        run_process_print('touch .env_setup_done')

up()  # this is needed always


#get_ipython().reset()  # run in ipython

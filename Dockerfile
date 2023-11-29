#FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
FROM tensorflow/tensorflow:2.1.0-gpu-py3
RUN apt-get update
RUN apt install -y wget
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

SHELL ["/bin/bash", "--login", "-c"]

RUN conda update -n base -c defaults conda
RUN conda install python=3.7
RUN conda create --name dabio python=3.7
RUN conda install --name dabio -c anaconda -y pandas==1.1.3 seaborn==0.11.0 scikit-learn==0.23.2 opencv imgaug==0.4.0 scikit-image==0.17.2 statsmodels==0.12.1 configparser==5.0.1 keras-gpu tensorflow-gpu==2.1
ENV PATH=/opt/conda/envs/dabio/bin:$PATH
ENV PATH=/opt/conda/bin:$PATH
RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64   
RUN touch /etc/ld.so.conf.d/cuda.conf

COPY dataset_train_reduced4.pickle /dataset_train_reduced4.pickle
COPY DA.py /DA.py
COPY EA.py /EA.py
COPY mainEA.py /mainEA.py

COPY states_seed29.pickle /home/states_seed29.pickle
COPY TestR4P_100G_Seed29.pickle /home/TestR4P_100G_Seed29.pickle
COPY TestR4P_100G_Seed29.csv /home/TestR4P_100G_Seed29.csv

#COPY states_seed27.pickle /home/states_seed27.pickle
#COPY TestR4P_100G_Seed27.pickle /home/TestR4P_100G_Seed27.pickle
#COPY TestR4P_100G_Seed27.csv /home/TestR4P_100G_Seed27.csv


SHELL ["conda", "run", "-n", "dabio", "/bin/bash", "-c"]
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "dabio", "python", "mainEA.py"]
#CMD ["python", "mainEA.py"]
#CMD ["python", "ScriptDA.py"]

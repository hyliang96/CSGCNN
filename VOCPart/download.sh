#!/usr/bin/env bash

# get absoltae path to the dir this is in, work in bash, zsh
# if you want transfer symbolic link to true path, just change `pwd` to `pwd -P`
here=$(cd "$(dirname "${BASH_SOURCE[0]-$0}")"; pwd)
scgcnn_voc_root="${here}/.."
cd ${scgcnn_voc_root}


while true; do
    dataset_dir=$(bash -c "read -p 'set a path to create the dir of VOC dataset? ' c; echo \$c");

    # if [[ ! "${dataset_dir}" =~ ^/ ]]; then
    #     echo "please input an absolute path"
    # else
    dataset_dir_parent=$(dirname ${dataset_dir})
    if [ -d "${dataset_dir_parent}" ]; then
        mkdir ${dataset_dir}
        [ $? = 0 ] && break
    else
        echo "no dir ${dataset_dir_parent}"
    fi
    # fi
done

echo dataset_dir ${dataset_dir}

if [ -d "${dataset_dir}" ]; then
    ln -sfT ${dataset_dir} __data__
else
    echo "no dir ${dataset_dir}, maybe failed to create it"
    exit 1
fi

mkdir __data__/raw

# download training/validation data, 1.3GB tar file
# homepage: http://host.robots.ox.ac.uk/pascal/VOC/voc2010/index.html
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar -P __data__
tar -xvf __data__/VOCtrainval_03-May-2010.tar -C __data__
mv __data__/VOCdevkit/VOC2010/ __data__/raw
rm __data__/VOCtrainval_03-May-2010.tar
rm -rf __data__/VOCdevkit

# homepage: http://roozbehm.info/pascal-parts/pascal-parts.html 78M tar.gz file
wget http://roozbehm.info/pascal-parts/trainval.tar.gz -P __data__
mkdir __data__/raw/metadata
tar -xzvf __data__/trainval.tar.gz -C __data__/raw/metadata
rm __data__/trainval.tar.gz

# release this variable in the end of file
unset -v here
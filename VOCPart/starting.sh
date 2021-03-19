# get absolute path to the dir this is in, work in bash, zsh
# if you want transfer symbolic link to true path, just change `pwd` to `pwd -P`
VOCPart_repo_root=$(cd "$(dirname "${BASH_SOURCE[0]-$0}")"; pwd)

# download checkpoints
# To get the checkpoints reported in our paper,
# download `{STD,CSG}.pt` from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/9e9943b6d87c4997b1e8/),
# and put them to  `./checkpoints/{STD,CSG}.pt`.
cd ${VOCPart_repo_root}/checkpoints
wget -c -t 0 https://cloud.tsinghua.edu.cn/f/49c44f295e394d17bb80/?dl=1 -O CSG.pt
wget -c -t 0 https://cloud.tsinghua.edu.cn/f/a1360d274a0942fdae77/?dl=1 -O STD.pt

# download preprocessed dataset
cd ${VOCPart_repo_root}
wget -c -t 0 https://cloud.tsinghua.edu.cn/f/8031d67f50f24e2aaaac/?dl=1 -O preprocessed_VOC_Part.zip
unzip preprocessed_VOC_Part.zip
rm preprocessed_VOC_Part.zip
ln -s preprocessed_VOC_Part __data__

# make __result__ link
while true; do
    result_dir=$(bash -c "read -p 'Path to result dir? (default=./result-dir/) ' c; echo \$c");
    if [ "${result_dir}" = '' ]; then
        result_dir='./result-dir'
    fi

    result_dir_parent="$(dirname "${result_dir}")"

    if [ -d "${result_dir_parent}"  ]; then
        mkdir -p "${result_dir}"
        break
    else
        echo "Dir not found: ${result_dir_parent}."
    fi
done

if [ -d "${result_dir}" ]; then
    ln -s "${result_dir}" __result__
else
    echo "Fail to make dir ${result_dir}."
fi

if [ ! -L __result__ ]; then
    echo "Fail to make soft link ./__result__ to ${result_dir}."
fi

# release this variable in the end of file
unset -v VOCPart_repo_root
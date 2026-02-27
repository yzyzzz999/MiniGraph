#!/bin/bash
# CN-DBpedia 中文知识图谱数据下载脚本
# 复旦大学知识工场实验室
# 包含 900万+ 实体，6700万+ 三元组

set -e

DATA_DIR="/autodl-fs/data/MiniGraph/data/raw"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "=============================================="
echo "MiniGraph - CN-DBpedia 数据下载"
echo "来源: 复旦大学知识工场实验室"
echo "开始时间: $(date)"
echo "数据目录: $DATA_DIR"
echo "=============================================="
echo ""
echo "数据集说明:"
echo "  - 900万+ 百科实体"
echo "  - 6700万+ 三元组关系"
echo "  - 110万+ mention2entity"
echo "  - 400万+ 摘要信息"
echo "  - 1980万+ 标签信息"
echo "  - 4100万+ infobox信息"
echo ""

# CN-DBpedia 下载地址
URL1="http://openkg.cn/dataset/cn-dbpedia"
URL2="https://pan.baidu.com/s/1eS1QT8E"  # 百度网盘

echo "----------------------------------------------"
echo "[$(date +%H:%M:%S)] 尝试从 OpenKG 下载..."
echo "URL: $URL1"
echo ""
echo "注意: CN-DBpedia 数据较大(约2-3GB)，下载可能需要较长时间"
echo "如果下载失败，可以手动从以下地址下载:"
echo "  1. OpenKG: http://openkg.cn/dataset/cn-dbpedia"
echo "  2. 百度网盘: https://pan.baidu.com/s/1eS1QT8E"
echo ""

# 尝试下载 (OpenKG 可能需要登录)
# 这里我们先创建说明文件
cat > README.txt <> "README"
CN-DBpedia 数据下载说明
========================

数据集信息:
  名称: CN-DBpedia
  来源: 复旦大学知识工场实验室
  官网: https://kw.fudan.edu.cn/cndbpedia/
  
数据规模:
  - 实体数量: 900万+
  - 三元组数量: 6700万+
  - mention2entity: 110万+
  - 摘要信息: 400万+
  - 标签信息: 1980万+
  - infobox信息: 4100万+

下载地址:
  1. OpenKG: http://openkg.cn/dataset/cn-dbpedia
  2. 百度网盘: https://pan.baidu.com/s/1eS1QT8E (提取码: 见官网)

数据格式:
  每行一个三元组: (实体名称, 属性名称, 属性值)
  字段之间用 tab 分隔
  
示例:
  复旦大学    简称    复旦
  复旦大学    地址    上海市杨浦区邯郸路220号

引用:
  @inproceedings{xu2017cn,
    title={CN-DBpedia: A Never-Ending Chinese Knowledge Extraction System},
    author={Xu, Bo and Xu, Yong and Liang, Jiaqing and Xie, Chenhao and Liang, Bin and Cui, Wanyun and Xiao, Yanghua},
    booktitle={International Conference on Industrial, Engineering and Other Applications of Applied Intelligent Systems},
    pages={428--438},
    year={2017},
    organization={Springer}
  }
README

echo "[$(date +%H:%M:%S)] 已创建下载说明文件: README.txt"
echo ""

echo "=============================================="
echo "CN-DBpedia 数据下载说明"
echo "=============================================="
echo ""
echo "由于 CN-DBpedia 数据需要通过 OpenKG 或百度网盘下载，"
echo "请手动下载后放置到以下目录:"
echo "  $DATA_DIR"
echo ""
echo "下载步骤:"
echo "  1. 访问 http://openkg.cn/dataset/cn-dbpedia"
echo "  2. 注册并登录 OpenKG 账号"
echo "  3. 下载 CN-DBpedia Dump 数据"
echo "  4. 将数据文件放置到: $DATA_DIR"
echo ""
echo "或者使用百度网盘:"
echo "  链接: https://pan.baidu.com/s/1eS1QT8E"
echo "  (提取码请查看官网)"
echo ""
echo "=============================================="

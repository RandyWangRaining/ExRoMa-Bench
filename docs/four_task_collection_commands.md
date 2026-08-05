# Four-Task Collection Commands

本文档集中记录 ExRoMa-Bench 当前四个核心任务的100次数据采集命令：

1. `stack_blocks_two`
2. `handover_block`
3. `beat_block_hammer`
4. `test_tube_rack`

所有命令均在月球大地图上执行 cuRobo 自动专家，开启严格碰撞检测，统计100次总尝试的成功率，并且只将成功轨迹保存为 HDF5。HDF5 内的三视角图像不添加文字标签；标签只会出现在可选的 MP4 预览中。

## 环境准备

```bash
cd /path/to/ExRoMa
conda activate exroma

./exroma.sh assets pull
./exroma.sh doctor
```

资产只需要在首次安装或资产版本发生变化时重新下载。已有本地资产目录时，也可以执行
`./exroma.sh assets link /path/to/ExRoMa-assets`，避免重复下载。

## 1. 叠放两个方块

```bash
TERM=xterm ./exroma.sh collect \
  --task stack_blocks_two \
  --scene procedural_moon \
  --attempts 100 \
  --seed 20000 \
  --strict-collision-check \
  --headless
```

默认输出目录：

```text
datasets/stack_blocks_two_procedural_moon_seed20000_run100/
```

## 2. 双臂交接方块

```bash
TERM=xterm ./exroma.sh collect \
  --task handover_block \
  --scene procedural_moon \
  --attempts 100 \
  --seed 30000 \
  --strict-collision-check \
  --headless
```

默认输出目录：

```text
datasets/handover_block_procedural_moon_seed30000_run100/
```

## 3. 抓锤敲击方块

```bash
TERM=xterm ./exroma.sh collect \
  --task beat_block_hammer \
  --scene procedural_moon \
  --attempts 100 \
  --seed 40000 \
  --strict-collision-check \
  --headless
```

默认输出目录：

```text
datasets/beat_block_hammer_procedural_moon_seed40000_run100/
```

## 4. 放置试管

```bash
TERM=xterm ./exroma.sh collect \
  --task test_tube_rack \
  --scene procedural_moon \
  --attempts 100 \
  --seed 50000 \
  --strict-collision-check \
  --headless
```

默认输出目录：

```text
datasets/test_tube_rack_procedural_moon_seed50000_run100/
```

## 连续采集四个任务

下面的命令依次运行四个任务。每项执行100次总尝试；前一个任务正常结束后才启动下一个任务。

```bash
declare -A seeds=(
  [stack_blocks_two]=20000
  [handover_block]=30000
  [beat_block_hammer]=40000
  [test_tube_rack]=50000
)

for task in stack_blocks_two handover_block beat_block_hammer test_tube_rack
do
  TERM=xterm ./exroma.sh collect \
    --task "$task" \
    --scene procedural_moon \
    --attempts 100 \
    --seed "${seeds[$task]}" \
    --strict-collision-check \
    --headless
done
```

## 查看成功率

每个输出目录中的 `collection_summary.json` 包含：

- `attempts`：已经执行的总尝试数；
- `successes`：成功并保存的轨迹数；
- `failures`：失败尝试数；
- `success_rate`：`successes / attempts`；
- `failure_counts`：按终止原因汇总的失败数量。

例如：

```bash
python -m json.tool \
  datasets/handover_block_procedural_moon_seed30000_run100/collection_summary.json
```

任务中断时，摘要会保留已完成尝试的统计，并将 `complete` 标记为 `false`。未成功的临时 HDF5 会被删除，不会混入训练数据。

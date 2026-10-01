# 火山引擎机器学习平台 CLI 提交训练任务指南

本文档说明如何在开发机 `daxiang` 上通过 `volc` 命令行工具（CLI）提交自定义训练任务，效果与火山引擎控制台网页端提交等效。

> **实测环境（2026-08-16）**
>
> - SSH 主机别名：`daxiang`
> - 登录用户：`haozhe.jia`
> - CLI 版本：`volc/1.2.57`
> - 地域：`cn-beijing`
> - 已验证：CLI 可以读取当前账号下的训练任务列表
> - 未验证：本文档更新过程中没有实际提交新任务，避免意外占用资源或产生费用

---

## 一、首次配置

### 0. 登录开发机

在自己的电脑终端中运行：

```bash
ssh daxiang
```

后续安装、配置和提交命令均在 `daxiang` 上执行。

### 1. 安装 volc CLI

```bash
sh -c "$(curl -fsSL https://ml-platform-public-examples-cn-beijing.tos-cn-beijing.volces.com/cli-binary/install.sh)" && export PATH=$HOME/.volc/bin:$PATH
```

安装后确认 `$HOME/.volc/bin` 在 `$PATH` 中。`daxiang` 使用 Bash，可将下面一行加入 `~/.bashrc`：

```bash
export PATH=$HOME/.volc/bin:$PATH
```

让当前会话立即生效：

```bash
source ~/.bashrc
```

验证安装：

```bash
command -v volc
volc version
```

实测应显示类似：

```text
/file_system/vepfs/intern/haozhe.jia/.volc/bin/volc
volc/1.2.57
```

### 2. 在火山引擎控制台创建 AK/SK

需要使用「大象未来」主账号下、具备机器学习平台权限的 IAM 子用户。不要使用没有相应权限的个人账号，也不建议使用主账号长期密钥。

1. 登录火山引擎控制台。
2. 确认当前身份是「大象未来」主账号下的 IAM 子用户。
3. 进入「访问控制」→「API 访问密钥」。
4. 点击「创建 Access Key」。
5. 完成短信验证。
6. 点击「查看密钥详情」或「下载凭证」，保存完整的 `AccessKeyId` 和 `SecretAccessKey`。

> **重要**
>
> - `SecretAccessKey` 可能以 `==` 结尾，末尾字符不能省略。
> - AK/SK 不要发到群聊、聊天机器人、工单正文或代码仓库。
> - 如果密钥已经泄露，应在控制台删除并重新创建。

### 3. 配置身份凭证（推荐使用文件方式）

在本次实测环境中，直接执行 `volc configure` 会先调用旧版校验接口，新创建的密钥可能被误报为：

```text
SK不合法：The request signature we calculated does not match...
```

但同一组 AK/SK 写入标准凭据文件后，`volc ml_task list` 可以正常鉴权。因此在 `daxiang` 上推荐直接配置文件。

先创建目录并限制权限：

```bash
mkdir -p ~/.volc
chmod 700 ~/.volc
```

使用 `vim ~/.volc/credentials` 或 `nano ~/.volc/credentials` 创建凭据文件：

```ini
[default]
access_key_id = <完整的 AccessKeyId>
secret_access_key = <完整的 SecretAccessKey>
```

再使用 `vim ~/.volc/config` 或 `nano ~/.volc/config` 创建地域配置：

```ini
[default]
region = cn-beijing
```

设置文件权限：

```bash
chmod 600 ~/.volc/credentials ~/.volc/config
```

确认文件存在和权限正确，但不要输出文件内容：

```bash
stat -c '%a %n' ~/.volc/credentials ~/.volc/config
```

预期结果：

```text
600 /你的HOME目录/.volc/credentials
600 /你的HOME目录/.volc/config
```

### 4. 验证鉴权

```bash
volc ml_task list
```

如果能看到任务列表，说明 CLI、AK/SK、地域及机器学习平台权限均已生效。列表界面常用按键：

- `j` / `k`：上下选择
- `Enter`：查看任务详情
- `e`：导出当前任务配置
- `r`：刷新
- `q`：退出

本次实测已成功读取到多个 `Running` 任务。

### 5. 常见配置问题

#### `volc: command not found`

```bash
export PATH="$HOME/.volc/bin:$PATH"
source ~/.bashrc
```

#### `exec: "mlp": executable file not found in $PATH`

`volc` 会调用同目录下的 `mlp`。不要只通过绝对路径执行 `~/.volc/bin/volc`，应先把整个目录加入 `PATH`：

```bash
export PATH="$HOME/.volc/bin:$PATH"
volc ml_task list
```

#### `AccessKeyId or SecretAccessKey is empty`

检查：

```bash
test -f ~/.volc/credentials
stat -c '%a %n' ~/.volc/credentials
```

凭据文件的字段名必须是：

```ini
[default]
access_key_id = <AK>
secret_access_key = <SK>
```

#### `volc configure` 报 `SK不合法` 或签名不匹配

先确认 AK/SK 没有多余空格、换行或缺少末尾的 `=`。如果控制台下载的完整凭证仍被 `volc configure` 拒绝，请不要自行对 AK/SK 做 Base64 解码，直接按本节的标准文件格式写入，再以 `volc ml_task list` 的实际结果为准。

#### 能鉴权但提示无权访问机器学习平台

说明密钥有效，但 IAM 子用户权限不足。需要联系「大象未来」账号管理员，为该 IAM 子用户授予机器学习平台任务、队列、镜像及相关存储的必要权限。

---

## 二、快速开始：从已有任务导出配置（推荐首次使用）

最稳妥的方式是先在网页端成功提交一次任务，然后导出它的配置作为模板：

```bash
# 1. 查看任务列表，找到一个已成功的任务 ID
volc ml_task list

# 2. 导出该任务的 YAML 配置文件
volc ml_task export --task <任务ID> --config

# 3. 导出的文件会保存在当前目录，基于它修改即可
```

任务 ID 格式形如 `t-20211216120106-vx7d4`，在网页端任务详情页的 URL 中也能看到。

选择模板时应优先使用与目标训练相近、已成功运行的任务。不要直接把入口命令为 `sleep inf` 的资源占位任务当作训练模板；如确需使用，必须把 `Entrypoint` 改为实际训练命令。

---

## 三、YAML 配置文件详解

### 完整配置模板

```yaml
# ========== 基础信息 ==========
TaskName: "my-training-task"           # 任务名称
Description: "训练任务描述"              # 可选
Tags:                                  # 可选标签
  - tag1
  - tag2

# ========== 入口命令 ==========
# 任务启动时执行的命令，相当于网页端的「入口命令」文本框
Entrypoint: "bash /root/code/start.sh"

# 入口命令的参数（会拼接到 Entrypoint 后面），可选
Args: ""

# ========== 代码上传 ==========
# 本地代码路径，提交时会自动上传到容器中
# 以 '/' 结尾：上传目录下的所有内容到 RemoteMountCodePath
# 不以 '/' 结尾：上传该目录本身及其下所有内容
UserCodePath: "./code"

# 容器中的代码挂载路径
RemoteMountCodePath: "/root/code/"

# ========== 环境变量 ==========
Envs:
  - Name: "RUN_NAME"
    Value: "experiment_001"
  - Name: "WANDB_API_KEY"
    Value: "your-key"
    IsPrivate: true     # 设为 true 后详情页仅创建人可见

# ========== 镜像 ==========
# 镜像 URL，在控制台「镜像中心」→ 选镜像 → 版本列表中复制
# 预置 PyTorch 镜像示例：
ImageUrl: "vemlp-cn-beijing.cr.volces.com/preset-images/pytorch:1.12.1"

# 如果使用私有仓库镜像，需要填写仓库凭证（预置镜像不需要）
# ImageCredential:
#   RegistryUsername: "xxx"
#   RegistryToken: "xxx"

# ========== 队列与资源 ==========
# 队列 ID：从控制台「资源队列」页面原样复制，不要根据队列名称猜测
ResourceQueueID: "<队列ID>"

# 当前 CLI 支持：TensorFlowPS / PyTorchDDP / MXNet / BytePS / MPI / Custom
Framework: "PyTorchDDP"

# 实例配置
TaskRoleSpecs:
  - RoleName: "worker"
    RoleReplicas: 1              # 实例数量（多机训练时 >1）
    Flavor: "ml.pmi2l"           # 实例规格，见下方说明
    # GpuRate: 1.0               # 可选，GPU 切分比例 (0,1]，整卡不用填

# 任务最长运行时间（秒），到时间自动停止
ActiveDeadlineSeconds: 432000    # 5天

# ========== 优先级与重试 ==========
Priority: 4                      # 优先级 1-9，默认支持 2/4/6
Preemptible: false               # 是否可抢占（闲时资源，可能被回收）

RetryOptions:
  EnableRetry: false
  MaxRetryTimes: 3
  IntervalSeconds: 300
  PolicySets:
    - "Failed"

# ========== 存储挂载 ==========
# 与网页端「共享文件系统」配置对应
Storages:
  # vePFS：代码与环境，必须挂载自己的子目录
  - Type: "Vepfs"
    MountPath: "/file_system/vepfs"
    SubPath: "users/your_name"   # 你的 vePFS 子目录
    ReadOnly: false

  # NAS：大文件 / checkpoint
  - Type: "Nas"
    MountPath: "/file_system/nas"
    # NasId 或 NasAddr 至少填一个，在 NAS 控制台查看
    NasId: "replace-with-nas-id"
    # NasAddr: "replace-with-nas-address"

  # EFS / TOS：训练数据
  - Type: "Tos"
    MountPath: "/file_system/efs"
    Bucket: "replace-with-bucket-name"
    # Prefix: "optional/sub/path"

# ========== 其他 ==========
EnableTensorBoard: false         # 是否开启 TensorBoard
AccessType: "Queue"              # 可见范围：Public / Queue / Private
```

### 常用实例规格（Flavor）

| Flavor | GPU | 说明 |
|--------|-----|------|
| `ml.pmi2l` | 1× NVIDIA A800-SXM4-80GB | 单卡 A800 |
| `ml.hpcpni2l` | 8× NVIDIA A800-SXM4-80GB | 8卡 A800 整机 |

> 完整规格列表见控制台或火山引擎「实例规格及定价」文档。提交前确认队列有对应资源。

---

## 四、入口命令的必要配置（重要！）

与网页端一样，入口命令或启动脚本中**必须包含**以下配置，否则会出现路径和权限问题：

```bash
#!/bin/bash
set -euo pipefail
umask 000

# 解决 /file_system 路径兼容问题
# 开发机上有 /file_system 目录，但火山云任务容器中可能没有
if [ ! -e /file_system ]; then
    ln -s / /file_system
fi

# 确保输出目录权限为 777（任务以 root 运行，普通用户需要读写权限）
OUTPUT_DIR="/file_system/nas/users/your_name/checkpoints/your_run_name"
mkdir -p "$OUTPUT_DIR"
chmod 777 "$OUTPUT_DIR"

# ===== 下面是你的实际训练命令 =====
cd /root/code
python train.py --output_dir "$OUTPUT_DIR"
```

将此脚本保存为 `start.sh`，放在代码目录中，YAML 的 `Entrypoint` 指向它：

```yaml
Entrypoint: "bash /root/code/start.sh"
```

---

## 五、提交任务

### 1. 提交前检查

在任务配置所在目录运行：

```bash
# CLI 和鉴权正常
volc version
volc ml_task list

# 配置文件和代码目录存在
test -f ./task_config.yaml
test -d ./code

# 启动脚本存在
test -f ./code/start.sh
```

还需要人工确认：

- `TaskName` 没有与不应覆盖或混淆的任务重名。
- `ImageUrl` 是完整镜像地址及正确 tag。
- `ResourceQueueID` 或队列名称来自控制台原值。
- `Flavor`、实例数及 GPU 数符合预期。
- `Entrypoint` 在容器中的路径正确。
- NAS、vePFS、TOS 的挂载信息和输出路径正确。
- 任务运行时长、优先级及可抢占设置符合预期。

### 2. 正式提交

```bash
# 基本提交
volc ml_task submit --conf=./task_config.yaml

# 等价的短参数
volc ml_task submit -c ./task_config.yaml

# 提交时临时覆盖某些参数
volc ml_task submit --conf=./task_config.yaml \
  --task_name "exp_002" \
  --entrypoint "bash /root/code/start.sh"

# 用 --set 覆盖任意配置项
volc ml_task submit --conf=./task_config.yaml \
  --set TaskRoleSpecs[0].RoleReplicas=2 \
  --set Priority=6

# 也可使用队列名称；当前 CLI 中队列名称优先于队列 ID
volc ml_task submit --conf=./task_config.yaml \
  --resource_queue_name "<控制台中的队列名称>"
```

提交成功后，任务会出现在网页端「自定义任务」列表中，可以在网页上查看日志、监控和 WebShell。

> **注意**：执行 `submit` 会创建真实训练任务、占用队列资源，并可能产生费用。提交前应再次确认 YAML 和命令行覆盖参数。

---

## 六、常用命令速查

```bash
# 查看任务列表
volc ml_task list

# 查看任务详情
volc ml_task get --id <任务ID>

# 查看任务日志
volc ml_task logs --task <任务ID> --instance worker_0

# 持续滚动日志（类似 tail -f）
volc ml_task logs --task <任务ID> --instance worker_0 -f

# 搜索日志中的错误
volc ml_task logs --task <任务ID> --instance worker_0 --content error

# 停止任务（在网页端操作或通过 API）
# 目前 CLI 没有直接的 stop 命令，需要在网页端停止

# 导出任务配置
volc ml_task export --task <任务ID> --config

# 导出任务代码
volc ml_task export --task <任务ID> --code
```

---

## 七、Slurm 风格提交（sbatch）

如果熟悉 Slurm，可以用更简洁的 `sbatch` 方式：

```bash
# 单节点 8 卡
volc ml_task sbatch \
  --nodes 1 \
  --partition <队列ID> \
  --gpus=8 \
  --image "vemlp-cn-beijing.cr.volces.com/preset-images/pytorch:1.12.1" \
  run.sh

# 多节点（2机16卡）
volc ml_task sbatch \
  --nodes 2 \
  --partition <队列ID> \
  --gpus=8 \
  run.sh
```

也可以在脚本头部用 `#SBATCH` 指令：

```bash
#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gres=gpu:8
#SBATCH --partition q-xxxxxxxxxxxxxx
#SBATCH --job-name my-experiment

set -euo pipefail
umask 000
if [ ! -e /file_system ]; then ln -s / /file_system; fi

python train.py
```

然后直接提交：

```bash
volc ml_task sbatch run.sh
```

> **注意**：sbatch 方式对存储挂载的控制不如 YAML 精细。如果需要挂载 vePFS/EFS/NAS，建议用 `--conf` 指定额外配置文件，或直接用 `submit` 方式。

---

## 八、注意事项

1. **AK/SK 安全**：不要把 AK/SK 硬编码在代码或 YAML 中，也不要提交到 Git。凭据保存在 `$HOME/.volc/credentials`，目录权限应为 `700`、文件权限应为 `600`。如果完整 AK/SK 曾出现在聊天、截图或日志中，应立即轮换。

2. **代码上传**：`UserCodePath` 指定的目录会在每次提交时上传。大文件（数据集、checkpoint）不要放在代码目录里，应放在 NAS/vePFS 中通过挂载访问。增量上传默认开启（`--local_diff on`）。

3. **队列 ID 不是队列名称**：从控制台「资源队列」页面复制准确值，不要手工猜测连字符或下划线格式。当前 CLI 也支持 `--resource_queue_name`，且队列名称参数优先于队列 ID。

4. **镜像 URL 要精确**：网页端选镜像只需要点选，CLI 需要填完整镜像 URL，包括版本 tag。在镜像中心详情页的「版本列表」中复制。

5. **存储路径一致性**：CLI 的 `MountPath` 和 `SubPath` 必须与网页端挂载配置一致，否则代码中写死的路径会找不到文件。

6. **首次使用建议**：先用网页端提交一个低资源、短时测试任务（如 `sleep 300`），导出配置确认各字段，再用 CLI 提交同样的配置验证等效性。不要使用 `sleep inf` 作为首次测试，以免忘记停止并持续占用资源。

7. **软链接处理**：如果代码目录中有软链接，提交时加 `--copy-links`（上传实际文件）或 `--links`（保持软链接），默认不上传软链接目标。

---

## 九、完整工作流示例

```bash
# 0. 一次性配置（详见“第一节：首次配置”）
# ~/.volc/credentials
# [default]
# access_key_id = <完整AK>
# secret_access_key = <完整SK>
#
# ~/.volc/config
# [default]
# region = cn-beijing

chmod 700 ~/.volc
chmod 600 ~/.volc/credentials ~/.volc/config
volc ml_task list

# 1. 准备代码目录
mkdir -p ./code
cp train.py start.sh ./code/

# 2. 编写配置
cat > task.yaml << 'EOF'
TaskName: "cli-test-task"
Entrypoint: "bash /root/code/start.sh"
UserCodePath: "./code/"
RemoteMountCodePath: "/root/code/"
ImageUrl: "vemlp-cn-beijing.cr.volces.com/preset-images/pytorch:1.12.1"
ResourceQueueID: "<从控制台原样复制的队列ID>"
Framework: "PyTorchDDP"
TaskRoleSpecs:
  - RoleName: "worker"
    RoleReplicas: 1
    Flavor: "ml.pmi2l"
ActiveDeadlineSeconds: 86400
Storages:
  - Type: "Vepfs"
    MountPath: "/file_system/vepfs"
    SubPath: "users/your_name"
  - Type: "Nas"
    MountPath: "/file_system/nas"
    NasId: "your-nas-id"
  - Type: "Tos"
    MountPath: "/file_system/efs"
    Bucket: "your-bucket"
EOF

# 3. 提交
volc ml_task submit --conf=task.yaml

# 4. 查看状态和日志
volc ml_task list
volc ml_task logs --task <返回的任务ID> --instance worker_0 -f
```

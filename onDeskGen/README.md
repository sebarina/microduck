# OnDeskGen

从**一段文字或一张图片**生成可以直接放进 MuJoCo 仿真的家具资产（沙发、书桌、椅子、茶几、边柜……）。

群核 Lux3D 负责生成和理解 3D，OnDeskGen 负责把结果变成仿真资产。最终得到：
- 米制、Z 朝上的模型；
- CoACD 凸分解的碰撞代理；
- 质量、惯量和摩擦；
- Lux3D 识别出的抽屉、柜门，转成 MuJoCo 的滑动 / 转动关节；
- MJCF 片段、资产包，以及给策略训练用的静态 MJCF。

## 流程

```
文字 ──文生图──▶ 概念图 ──图生 3D──▶ 带贴图的米制模型 ──部件拆分──▶ 部件 ──铰接──▶ URDF（可动部件 + 关节）
文字 ──文生 3D────────────────────▶ 模型
图片 ──四视图──▶ 4 个视角 ──多图生 3D──▶ 模型
模型 + 照片 ──材质迁移──▶ 换成照片里的材质
模型 + 「图 + 文字」改出的参考图 ──材质迁移──▶ 同一外形、新材质
模型 ──多格式导出──▶ USDZ / OBJ / STL
        │
        ▼
归一化（米制、Z 朝上、原点在底面中心、正面朝 +X）→ 视觉网格（减面、按颜色分组）
→ CoACD 凸包（每个可动部件单独分解；离地不到 1 cm 的腿自动调平）
→ 质量与惯量（按类别参考质量，按凸包体积分配给主体和各部件）→ 摩擦与接触参数
→ MJCF 片段（自由刚体 + 关节）→ 网页预览 glb
```

每个 Lux3D 任务都会归档到 `assets/raw/<资产>/<功能>/<任务号>/`，包括输入、输出和哈希。同样的请求再跑一次直接用归档，不花积分。

## 安装

需要 Python 3.12 和 [uv](https://docs.astral.sh/uv/)。

```bash
cd onDeskGen
uv sync
```

## 配置

| 环境变量 | 作用 |
|---|---|
| `AHOLO_API_KEY` | Lux3D API key（必需，用于创建新任务） |
| `AHOLO_REGION` | Lux3D 区域，默认 `cn` |
| `ONDESKGEN_UPLOAD_URL` | 文件服务：把本地模型 / 图片交给 Lux3D 时需要 URL。要求是 HTTP PUT 接口，`?fileName=...`，返回 `{"code": 200, "data": {"fileUrl": "https://..."}}`，例如用 Cloudflare Worker 接一个 R2 桶。部件拆分、铰接、导出都要用到。 |
| `ONDESKGEN_HOME` | 数据目录，默认是本项目目录 |
| `ONDESKGEN_CONFIG` | 引擎配置文件，默认 `configs/ondeskgen.json` |

材质迁移的参考图走 Aholo 官方上传（`manycore-aholo-sdk-asset`），用的是同一个 `AHOLO_API_KEY`。

`configs/ondeskgen.json` 把所有参数集中在一处：
- Lux3D 选项和积分上限 `credit_cap`；
- CoACD 参数；
- 摩擦和接触参数；
- 每个类别的默认尺寸、合理尺寸范围、正面判定规则、参考质量、域随机化范围和提示词模板。

每个资产都会记录这份配置的哈希。

## 使用

```bash
# 文字 → 概念图 → 3D → 仿真资产（书桌会自动做部件拆分和抽屉铰接）
uv run ondeskgen generate desk --text "浅色橡木书桌，右侧带两个抽屉"

# 其他输入方式
uv run ondeskgen generate sofa --text "深灰色布艺三人沙发" --no-concept      # 文字直接生成 3D
uv run ondeskgen generate chair --image ./chair.jpg --material               # 照片 → 3D → 材质迁移
uv run ondeskgen generate desk --image ./desk.jpg --four-view                # 单图 → 四视图 → 3D
uv run ondeskgen generate sofa --image ./sofa.jpg --restyle "换成墨绿色丝绒"  # 同一外形换材质
uv run ondeskgen generate cabinet --text "两扇对开柜门的边柜" \
    --articulation-prompt "正面的两扇柜门可以绕竖直的铰链转动打开和关上"         # 柜门 → 转动关节
uv run ondeskgen generate desk --text "..." --size 1.4 0.7 0.75 --final      # 指定尺寸（宽 深 高，米），用 G1
uv run ondeskgen generate desk --text "..." --dry-run                        # 只看计划，不花积分

# 打包
uv run ondeskgen pack my_pack --all          # MuJoCo 资产包（+ zip）
uv run ondeskgen train-export --all          # 训练用静态 MJCF（box / mesh 两种碰撞）
```

## 输出

```
assets/processed/<资产>/
  <资产>.xml          MJCF 片段：自由刚体（+ 可动部件的关节）
  visual/**.obj       视觉网格（不参与碰撞）
  collision/*.obj     CoACD 凸包
  asset.json          归一化、碰撞、物理、关节、来源的完整记录
  ondeskgen.json      简要标记（类别、提示词、用到的 Lux3D 功能）
outputs/ondeskgen/runs/<资产>/   run.json（每一步的任务号、耗时、积分）、events.jsonl、preview.glb
outputs/packs/<名字>/             资产包：assets/、scenes/（每个资产一个场景 + 全部并排的 showroom）、pack.json、README
outputs/train_export/<名字>/      静态 MJCF：box/、mesh/，每个资产一个目录，附 manifest 和 README
```

**训练用静态 MJCF**：没有 freejoint、关节和 inertial，抽屉、柜门固定在关闭位置；原点在底面中心；网格都是相对路径的 `.obj`；名字都带资产 id 前缀，不用 `<default>` 类，方便合并进训练场景。提供两种碰撞：
- `box`：每个凸包换成一个有向包围盒，贴地的一律竖直放置，GPU 仿真最快；
- `mesh`：每个凸包简化到最多 64 个顶点。

加载资产包：

```bash
uv run python -m mujoco.viewer --mjcf=outputs/packs/my_pack/scenes/showroom.xml
```

## 注意

- **质量和摩擦是估计值**：按类别的参考值设定，Lux3D 不提供材质和质量。资产包的 `physics.json` 里给了训练用的随机化范围。
- **碰撞块按几何形状切分**（CoACD 凸分解），不是按坐垫、扶手这样的语义部件；但凹处（扶手之间、桌下）会保留。
- **铰接依赖 Lux3D 能否识别出可动部件**：识别不出时，资产按没有关节的整体处理。
- **生成内容的许可**：家具模型由 Lux3D 生成，使用前请确认 Lux3D 服务条款对生成内容的约定。

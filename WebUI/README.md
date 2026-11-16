# OnDeskGen Web UI

OnDeskGen 的网页界面，顶部导航三个页面：

| 页面 | 地址 | 内容 |
|---|---|---|
| 演示视频 | `/` | 整体演示视频，右侧按章节跳转 |
| 资产引擎 | `/assets` | 选类别、输入文字或上传图片、勾选要用的 Lux3D 功能，现场调用 OnDeskGen 生成，实时看每一步；资产库里每个资产都能 3D 预览（拖动旋转、拉开抽屉 / 柜门、显示碰撞凸包），勾选后打包下载 |
| 仿真训练 | `/train` | 用 OnDeskGen 资产训练 MicroDuck 在家具表面行走：链路、训练用资产、云端训练流程、训练记录 |

服务端只用 Python 标准库，只监听本机。前端用 three.js 0.160.0 和 Chakra Petch 字体，都已放在 `src/webui/static/vendor/`，各自附带许可证。

## 安装和启动

WebUI 依赖同级目录的 `onDeskGen`，目录结构要保持：

```
<仓库>/
  onDeskGen/
  WebUI/
```

```bash
cd WebUI
uv sync
uv run ondeskgen-webui            # http://127.0.0.1:8765，--port 可改端口
```

资产引擎要现场生成，需要在启动前设置好 OnDeskGen 的环境变量（`AHOLO_API_KEY`、`ONDESKGEN_UPLOAD_URL` 等，见 `../onDeskGen/README.md`）。不设置也能浏览已经生成好的资产。页面只知道 key 有没有设置，不会拿到 key 本身。

## 数据从哪里来

| 页面 | 数据 |
|---|---|
| 资产引擎 | OnDeskGen 的数据目录（`ONDESKGEN_HOME`，默认 `../onDeskGen`）：`outputs/ondeskgen/runs/`、`assets/processed/` |
| 演示视频 | `media/demo.mp4`，以及可选的章节文件 `media/demo_chapters.json`（格式 `{"duration_s": 237.3, "chapters": [{"t": 6.0, "title": "资产生成", "kicker": "1 / 4", "lines": ["..."]}]}`） |
| 仿真训练 | 页面文字在 `configs/sim_train.json`；训练用资产来自 OnDeskGen 的 `outputs/train_export/ondeskgen_static_v1/`；可选：训练侧检查输出 `media/sim_train/checks/<名字>.txt`（最后一行是 `exit 0` 表示通过），策略在资产上运行的结果 `media/sim_train/slots/<名字>/slot.json`（附视频） |

`media/` 里的文件不进 git，视频请另行放置。

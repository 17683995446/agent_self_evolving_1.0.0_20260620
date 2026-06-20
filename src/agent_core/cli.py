"""
统一命令行接口。

提供单一入口控制Agent全部行为。

第一性原理：
- 生物运动皮层统一指挥肢体运动——单一决策中枢协调全部行为
- CLI是用户与系统交互的"运动皮层"，所有操作通过统一入口
- 子命令模式：每个行为一个明确命令，降低认知负荷

设计要点：
- argparse子命令：start/stop/status/health/dashboard/log/version
- 零配置可启动：无参数时使用合理默认值
- 信号处理：Ctrl+C优雅关闭
- 友好输出：彩色状态、进度提示
"""

import argparse
import asyncio
import json
import os
import signal
import sys
import time
from typing import Any, Dict, List, Optional

from agent_core.common.logger import get_logger
from agent_core.common.constants import LogLevel

# 默认配置
_DEFAULT_ONLINE_INTERVAL = 1.0
_DEFAULT_OFFLINE_INTERVAL = 3600.0
_DEFAULT_DASHBOARD_PORT = 8080
_DEFAULT_ENV = "dev"

# 输出颜色（ANSI转义码）
_COLOR_RESET = "\033[0m"
_COLOR_RED = "\033[31m"
_COLOR_GREEN = "\033[32m"
_COLOR_YELLOW = "\033[33m"
_COLOR_BLUE = "\033[34m"
_COLOR_CYAN = "\033[36m"
_COLOR_BOLD = "\033[1m"

# 是否启用颜色（检测TTY）
_COLOR_ENABLED = sys.stdout.isatty() if hasattr(sys.stdout, "isatty") else False


def _colorize(text: str, color: str) -> str:
    """添加颜色（非TTY环境不添加）。"""
    if not _COLOR_ENABLED:
        return text
    return f"{color}{text}{_COLOR_RESET}"


def _print_success(msg: str) -> None:
    """打印成功信息。"""
    print(f"{_colorize('[OK]', _COLOR_GREEN)} {msg}")


def _print_error(msg: str) -> None:
    """打印错误信息。"""
    print(f"{_colorize('[ERROR]', _COLOR_RED)} {msg}", file=sys.stderr)


def _print_warning(msg: str) -> None:
    """打印警告信息。"""
    print(f"{_colorize('[WARN]', _COLOR_YELLOW)} {msg}")


def _print_info(msg: str) -> None:
    """打印信息。"""
    print(f"{_colorize('[INFO]', _COLOR_BLUE)} {msg}")


def _load_config(env: str, config_path: Optional[str]) -> Dict[str, Any]:
    """
    加载配置。

    优先级：config_path > env配置 > base配置 > 默认值

    Args:
        env: 环境名（dev/prod/test）。
        config_path: 显式配置文件路径。

    Returns:
        配置字典。
    """
    from agent_core.common.config_validator import ConfigValidator

    config: Dict[str, Any] = {}

    if config_path and os.path.exists(config_path):
        # 显式配置文件优先
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            _print_info(f"Loaded config from {config_path}")
        except ImportError:
            _print_warning("yaml not available, using defaults")
        except OSError as exc:
            _print_error(f"Failed to load config: {exc}")
    else:
        # 使用EnvironmentConfig加载base + env配置
        config_dir = os.environ.get(
            "AGENT_CONFIG_DIR",
            os.path.join(os.getcwd(), "config"),
        )
        try:
            from agent_core.common.environment_config import EnvironmentConfig
            env_config = EnvironmentConfig(config_dir=config_dir)
            # 强制使用指定环境
            if env != env_config.environment:
                # 重新加载指定环境
                os.environ["AGENT_ENV"] = env
                env_config = EnvironmentConfig(config_dir=config_dir)
            config = env_config.all_config
            if config:
                _print_info(f"Loaded {env} config from {config_dir}")
        except Exception as exc:
            _print_warning(f"EnvironmentConfig load failed: {exc}, using defaults")

    # 填充默认值（零配置可启动）
    defaults = _get_default_config()
    for key, value in defaults.items():
        if key not in config:
            config[key] = value

    # 校验配置
    validator = ConfigValidator()
    ok, errors = validator.validate(config)
    if not ok:
        _print_error("Configuration validation failed:")
        for err in errors:
            _print_error(f"  {err}")
        _print_error("Fix the above errors or use --config to specify a valid config file")
        _print_info("Hint: Use 'agent start --dry-run' to check config without starting")
        _print_info("Hint: Use 'agent start' without --config for zero-config defaults")

    return config


def _get_default_config() -> Dict[str, Any]:
    """获取默认配置（零配置启动用）。"""
    return {
        "app_name": "self_evolving_agent",
        "app_version": "1.0.0",
        "log_level": "INFO",
        "online_interval_seconds": _DEFAULT_ONLINE_INTERVAL,
        "offline_interval_seconds": _DEFAULT_OFFLINE_INTERVAL,
        "memory_capacity": 10000,
        "exploration_rate": 0.3,
        "epsilon": 0.2,
        "interrupt_threshold": 0.7,
        "feature_dimensions": 10,
        "low_alpha_init": 0.3,
        "low_alpha_min": 0.05,
        "low_alpha_max": 0.8,
        "low_adaptation_rate": 0.1,
        "mid_window_size": 20,
        "mid_forecast_horizon": 5,
        "high_window_size": 100,
        "high_reversion_strength": 0.3,
        "ensemble_beta": 5.0,
        "cpu_interval_seconds": 1.0,
        "disk_paths": ["/"],
        "log_paths": [],
        "max_log_lines": 100,
        "base_weights": {
            "safety": 0.4,
            "gain": 0.4,
            "efficiency": 0.2,
        },
        "query_page_size": 20,
        "batch_query_limit": 1000,
    }


def _check_first_run(args: argparse.Namespace) -> None:
    """首次运行检测：无配置文件时提示运行 init。

    第一性原理：降低上手阻力——在用户最需要帮助的时刻（首次启动）
    主动引导，而非让用户在默认配置中摸索。
    """
    # 显式指定配置文件时不检测
    if args.config:
        return

    config_dir = os.environ.get(
        "AGENT_CONFIG_DIR",
        os.path.join(os.getcwd(), "config"),
    )
    env_config = os.path.join(config_dir, f"{args.env}.yaml")
    base_config = os.path.join(config_dir, "base.yaml")

    # 如果环境配置和base配置都不存在，视为首次运行
    if not os.path.exists(env_config) and not os.path.exists(base_config):
        _print_info("First run detected — no configuration files found.")
        _print_info(
            f"Tip: Run 'agent init --env {args.env}' to create a config file, "
            "or just proceed with built-in defaults."
        )


async def _cmd_start(args: argparse.Namespace) -> int:
    """启动Agent。"""
    from agent_core.memory.agent_main import SelfEvolvingAgent

    # 首次运行检测：无配置文件时提示运行 init
    _check_first_run(args)

    config = _load_config(args.env, args.config)

    # CLI参数覆盖配置
    if args.log_level:
        config["log_level"] = args.log_level
    if args.online_interval:
        config["online_interval_seconds"] = args.online_interval
    if args.offline_interval:
        config["offline_interval_seconds"] = args.offline_interval

    # --dry-run：仅验证配置不启动
    if args.dry_run:
        print(_colorize("=== Configuration Check ===", _COLOR_BOLD))
        print(f"  Environment:        {args.env}")
        print(f"  Log Level:          {config.get('log_level', 'INFO')}")
        print(f"  Online Interval:    {config.get('online_interval_seconds', 1.0)}s")
        print(f"  Offline Interval:   {config.get('offline_interval_seconds', 3600.0)}s")
        print(f"  Memory Capacity:    {config.get('memory_capacity', 10000)}")
        print(f"  Exploration Rate:   {config.get('exploration_rate', 0.3)}")
        _print_success("Configuration is valid. Agent is ready to start.")
        _print_info("Run without --dry-run to actually start the Agent.")
        return 0

    _print_info(
        f"Starting Agent (env={args.env}, "
        f"online={config['online_interval_seconds']}s, "
        f"offline={config['offline_interval_seconds']}s)"
    )

    agent = SelfEvolvingAgent()

    # 初始化（带进度反馈）
    print(_colorize("=== Startup Progress ===", _COLOR_BOLD))
    print(f"  {_colorize('[1/3]', _COLOR_CYAN)} Initializing Agent...")
    start_time = time.time()
    try:
        ok = await agent.initialize(config)
        if not ok:
            _print_error("Agent initialization failed")
            return 1
    except Exception as exc:
        _print_error(f"Initialization error: {exc}")
        return 1

    init_duration = time.time() - start_time
    print(f"  {_colorize('[OK]', _COLOR_GREEN)} Initialized in {init_duration:.2f}s")

    # 信号处理
    stop_event = asyncio.Event()

    def _signal_handler(signum, frame):
        _print_info(f"\nReceived signal {signum}, shutting down...")
        stop_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # 启动
    online_interval = config["online_interval_seconds"]
    offline_interval = config["offline_interval_seconds"]

    try:
        print(f"  {_colorize('[2/3]', _COLOR_CYAN)} Starting Agent loops...")
        await agent.start(online_interval, offline_interval)
        print(f"  {_colorize('[OK]', _COLOR_GREEN)} Agent loops started")
        print(f"  {_colorize('[3/3]', _COLOR_CYAN)} Dashboard at http://127.0.0.1:{_DEFAULT_DASHBOARD_PORT}")
        print()
        _print_success("Agent is running. Press Ctrl+C to stop.")
        _print_info(f"  Status:  agent status")
        _print_info(f"  Logs:    agent log --lines 50")
        _print_info(f"  Stop:    agent stop")

        # 等待停止信号
        await stop_event.wait()

    except KeyboardInterrupt:
        pass
    except Exception as exc:
        _print_error(f"Runtime error: {exc}")
        return 1
    finally:
        _print_info("Stopping Agent...")
        stop_start = time.time()
        try:
            await agent.stop()
            stop_duration = time.time() - stop_start
            _print_success(f"Agent stopped gracefully (in {stop_duration:.2f}s)")
        except Exception as exc:
            _print_error(f"Error during shutdown: {exc}")

    return 0


async def _cmd_status(args: argparse.Namespace) -> int:
    """查询Agent状态（通过Dashboard API或日志文件）。"""
    import urllib.request
    import urllib.error

    url = f"http://127.0.0.1:{args.port}/api/status"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError:
        # Dashboard不可用 — 离线模式
        _print_warning(
            f"Agent is not running (dashboard at port {args.port} not reachable)"
        )
        _print_info("Recent log entries:")

        # 尝试读取最近的日志文件
        log_dir = os.path.join(os.getcwd(), "data", "logs")
        log_file = os.path.join(log_dir, "selfevolvingagent.log")
        if os.path.exists(log_file):
            try:
                with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                # 显示最后10行
                recent = lines[-10:] if len(lines) > 10 else lines
                for line in recent:
                    print(f"  {line.rstrip()}")
                _print_info(f"(showing last {len(recent)} lines from {log_file})")
            except OSError as exc:
                _print_error(f"Failed to read log file: {exc}")
        else:
            _print_info("No log file found. Start the Agent with 'agent start'.")

        _print_info("To start the Agent: agent start")
        return 1
    except Exception as exc:
        _print_error(f"Status query failed: {exc}")
        return 1

    # 格式化输出
    print(_colorize("=== Agent Status ===", _COLOR_BOLD))
    status = data.get("status", "unknown")
    status_color = _COLOR_GREEN if status == "running" else _COLOR_YELLOW
    print(f"  Status:       {_colorize(status, status_color)}")

    # 运行时长（如果提供了启动时间）
    start_time = data.get("start_time")
    if start_time:
        try:
            # 支持ISO格式或时间戳
            if isinstance(start_time, (int, float)):
                elapsed = time.time() - start_time
            else:
                from datetime import datetime
                dt = datetime.fromisoformat(start_time)
                elapsed = time.time() - dt.timestamp()
            print(f"  Uptime:       {_format_duration(elapsed)}")
        except (ValueError, OSError):
            print(f"  Started:      {start_time}")

    print(f"  Cycle Count:  {data.get('cycle_count', 0)}")
    print(f"  Interrupts:   {data.get('interrupt_count', 0)}")

    # 吞吐率（每秒循环数）
    cycle_count = data.get("cycle_count", 0)
    if start_time and cycle_count > 0:
        try:
            if isinstance(start_time, (int, float)):
                elapsed = time.time() - start_time
            else:
                from datetime import datetime
                dt = datetime.fromisoformat(start_time)
                elapsed = time.time() - dt.timestamp()
            if elapsed > 0:
                rate = cycle_count / elapsed
                print(f"  Throughput:   {rate:.2f} cycles/sec")
        except (ValueError, OSError):
            pass

    # 任务统计
    tasks_total = data.get("tasks_total")
    tasks_completed = data.get("tasks_completed")
    if tasks_total is not None:
        print(f"  Tasks:        {tasks_completed or 0}/{tasks_total} completed")
        if tasks_total > 0:
            progress = (tasks_completed or 0) / tasks_total * 100
            bar = _progress_bar(progress)
            print(f"                {bar} {progress:.1f}%")

    errors = data.get("recent_errors", [])
    if errors:
        print(f"  Recent Errors ({len(errors)}):")
        for err in errors[-3:]:
            print(f"    - {err}")

    rewards = data.get("recent_rewards", [])
    if rewards:
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        print(f"  Avg Reward:   {avg_reward:.3f} (last {len(rewards)})")

    task = data.get("current_task")
    if task:
        print(f"  Current Task: {task}")

    return 0


def _format_duration(seconds: float) -> str:
    """格式化时长为人类可读形式。"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    elif seconds < 86400:
        return f"{seconds/3600:.1f}h"
    else:
        return f"{seconds/86400:.1f}d"


def _progress_bar(percent: float, width: int = 20) -> str:
    """生成文本进度条。"""
    filled = int(width * percent / 100)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}]"


async def _cmd_stop(args: argparse.Namespace) -> int:
    """停止运行中的Agent。

    优先通过Dashboard API发送停止请求；
    若Dashboard不可用，回退到查找并终止Agent进程。
    """
    import urllib.request
    import urllib.error
    import subprocess

    # 方式1：通过Dashboard API停止
    url = f"http://127.0.0.1:{args.port}/api/stop"
    try:
        req = urllib.request.Request(url, method="POST")
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                _print_success("Agent stop request sent via Dashboard API")
                return 0
    except urllib.error.URLError:
        pass  # Dashboard不可用，回退到进程查找
    except Exception:
        pass  # 其他API错误，回退到进程查找

    # 方式2：查找并终止Agent进程
    _print_info("Dashboard not reachable, trying to find Agent process...")
    try:
        result = subprocess.run(
            ["pgrep", "-f", "agent_core"],
            capture_output=True, text=True, timeout=5,
        )
        pids = result.stdout.strip().split()
        if not pids:
            _print_warning("No running Agent process found")
            return 0

        for pid_str in pids:
            pid = int(pid_str)
            if pid == os.getpid():
                continue  # 不终止自己
            try:
                os.kill(pid, signal.SIGTERM)
                _print_info(f"Sent SIGTERM to PID {pid}")
            except ProcessLookupError:
                pass
            except PermissionError:
                _print_error(f"Permission denied to stop PID {pid}")

        _print_success("Agent stop signal sent")
        return 0
    except FileNotFoundError:
        _print_error("pgrep not available. Use 'kill <pid>' manually.")
        return 1
    except Exception as exc:
        _print_error(f"Failed to stop Agent: {exc}")
        return 1


async def _cmd_health(args: argparse.Namespace) -> int:
    """健康检查。"""
    from agent_core.memory.agent_main import SelfEvolvingAgent

    _print_info("Running health check...")
    agent = SelfEvolvingAgent()
    config = _load_config(args.env, args.config)

    try:
        await agent.initialize(config)
        result = await agent.health_check()
    except Exception as exc:
        _print_error(f"Health check failed: {exc}")
        return 1

    try:
        await agent.stop()
    except Exception:
        pass

    print(_colorize("=== Health Check ===", _COLOR_BOLD))
    overall = result.get("status", "unknown")
    color = _COLOR_GREEN if overall == "healthy" else _COLOR_RED
    print(f"  Overall: {_colorize(overall.upper(), color)}")

    checks = result.get("checks", {})
    for name, check in checks.items():
        status = check.get("status", "unknown") if isinstance(check, dict) else str(check)
        icon = _colorize("[OK]", _COLOR_GREEN) if status == "healthy" else _colorize("[FAIL]", _COLOR_RED)
        print(f"  {icon} {name}: {status}")

    return 0 if overall == "healthy" else 1


async def _cmd_dashboard(args: argparse.Namespace) -> int:
    """启动独立Dashboard服务。"""
    from agent_core.monitoring.dashboard_server import get_dashboard_server

    dashboard = get_dashboard_server()
    _print_info(f"Starting dashboard on port {args.port}...")
    dashboard.start(port=args.port)
    _print_success(f"Dashboard running at http://127.0.0.1:{args.port}")
    _print_info("Press Ctrl+C to stop")

    # 等待信号
    stop_event = asyncio.Event()

    def _signal_handler(signum, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    await stop_event.wait()
    dashboard.stop()
    _print_success("Dashboard stopped")
    return 0


async def _cmd_version(args: argparse.Namespace) -> int:
    """显示版本信息。"""
    from agent_core import __version__
    print(_colorize("Self-Evolving Agent", _COLOR_BOLD))
    print(f"  Version: {__version__}")
    print(f"  Python:  {sys.version.split()[0]}")
    print(f"  Path:    {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")
    return 0


async def _cmd_log(args: argparse.Namespace) -> int:
    """查看最近日志。"""
    log_dir = os.path.join(os.getcwd(), "data", "logs")
    log_file = os.path.join(log_dir, "selfevolvingagent.log")

    if not os.path.exists(log_file):
        _print_warning("No log file found.")
        _print_info("Start the Agent with 'agent start' to generate logs.")
        return 1

    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError as exc:
        _print_error(f"Failed to read log file: {exc}")
        return 1

    n = args.lines
    recent = lines[-n:] if len(lines) > n else lines

    print(_colorize(f"=== Last {len(recent)} log lines ===", _COLOR_BOLD))
    print(f"  (from {log_file})")
    print()
    for line in recent:
        print(line.rstrip())

    return 0


async def _cmd_init(args: argparse.Namespace) -> int:
    """交互式生成配置文件。

    第一性原理：降低上手阻力——用户无需知道配置文件格式，
    通过问答即可生成可用配置。零配置到可用配置的桥梁。
    """
    print(_colorize("=== Self-Evolving Agent Configuration Wizard ===", _COLOR_BOLD))
    print()

    # 默认值
    defaults = _get_default_config()

    # 交互式收集关键参数（非交互模式直接用默认值）
    if not args.non_interactive:
        config = _interactive_collect(args, defaults)
    else:
        config = dict(defaults)
        config["log_level"] = args.log_level or defaults["log_level"]
        config["online_interval_seconds"] = args.online_interval or defaults["online_interval_seconds"]
        config["offline_interval_seconds"] = args.offline_interval or defaults["offline_interval_seconds"]
        config["memory_capacity"] = args.memory_capacity or defaults["memory_capacity"]
        config["exploration_rate"] = args.exploration_rate or defaults["exploration_rate"]

    # 确定输出路径
    env = args.env
    if args.output:
        output_path = args.output
    else:
        config_dir = os.path.join(os.getcwd(), "config")
        os.makedirs(config_dir, exist_ok=True)
        output_path = os.path.join(config_dir, f"{env}.yaml")

    # 检查文件是否已存在
    if os.path.exists(output_path) and not args.force:
        _print_warning(f"Configuration file already exists: {output_path}")
        _print_info("Use --force to overwrite, or specify --output for a different path")
        return 1

    # 生成配置文件（带注释）
    content = _generate_config_yaml(config, env)
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
    except OSError as exc:
        _print_error(f"Failed to write config file: {exc}")
        return 1

    _print_success(f"Configuration file created: {output_path}")
    print()
    print(_colorize("Next steps:", _COLOR_BOLD))
    print(f"  1. Review/edit: {output_path}")
    print(f"  2. Validate:    agent start --env {env} --dry-run")
    print(f"  3. Start:       agent start --env {env}")
    return 0


def _interactive_collect(
    args: argparse.Namespace, defaults: Dict[str, Any]
) -> Dict[str, Any]:
    """交互式收集配置参数。

    渐进式设计：只问最关键的5个参数，其余用默认值。
    """
    config = dict(defaults)

    def _ask(prompt: str, default: str) -> str:
        """询问用户输入，支持默认值。"""
        try:
            val = input(f"{prompt} [{default}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            return default
        return val if val else default

    _print_info("Answer the following questions (press Enter to accept defaults):")
    print()

    # 1. 日志级别
    log_level = args.log_level or _ask(
        "Log level (DEBUG/INFO/WARNING/ERROR)",
        defaults["log_level"],
    )
    config["log_level"] = log_level.upper()

    # 2. 在线循环间隔
    online_str = _ask(
        "Online loop interval (seconds)",
        str(defaults["online_interval_seconds"]),
    )
    try:
        config["online_interval_seconds"] = float(online_str)
    except ValueError:
        _print_warning(f"Invalid number '{online_str}', using default")
        config["online_interval_seconds"] = defaults["online_interval_seconds"]

    # 3. 离线循环间隔
    offline_str = _ask(
        "Offline loop interval (seconds)",
        str(defaults["offline_interval_seconds"]),
    )
    try:
        config["offline_interval_seconds"] = float(offline_str)
    except ValueError:
        _print_warning(f"Invalid number '{offline_str}', using default")
        config["offline_interval_seconds"] = defaults["offline_interval_seconds"]

    # 4. 记忆容量
    mem_str = _ask(
        "Memory capacity (number of experiences)",
        str(defaults["memory_capacity"]),
    )
    try:
        config["memory_capacity"] = int(mem_str)
    except ValueError:
        _print_warning(f"Invalid number '{mem_str}', using default")
        config["memory_capacity"] = defaults["memory_capacity"]

    # 5. 探索率
    explore_str = _ask(
        "Exploration rate (0.0-1.0, higher=more exploration)",
        str(defaults["exploration_rate"]),
    )
    try:
        rate = float(explore_str)
        if 0.0 <= rate <= 1.0:
            config["exploration_rate"] = rate
        else:
            _print_warning(f"Rate {rate} out of [0,1], using default")
    except ValueError:
        _print_warning(f"Invalid number '{explore_str}', using default")

    return config


def _generate_config_yaml(config: Dict[str, Any], env: str) -> str:
    """生成带注释的配置文件内容。

    第一性原理：配置文件应自解释——每项都有注释说明用途，
    降低用户理解成本。
    """
    lines = [
        f"# {env} 环境配置文件",
        f"# 由 'agent init' 生成于 {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"# 修改后用 'agent start --env {env} --dry-run' 验证",
        "",
        f"app_name: \"{config['app_name']}\"",
        f"app_version: \"{config['app_version']}\"",
        "",
        "# 日志级别: DEBUG/INFO/WARNING/ERROR/CRITICAL",
        f"log_level: \"{config['log_level']}\"",
        "",
        "# 运行循环间隔（秒）",
        "# online: 在线学习循环间隔（感知-决策-行动周期）",
        f"online_interval_seconds: {config['online_interval_seconds']}",
        "# offline: 离线优化循环间隔（经验回放与模型压缩）",
        f"offline_interval_seconds: {config['offline_interval_seconds']}",
        "",
        "# 记忆系统容量（存储的经验轨迹数量上限）",
        f"memory_capacity: {config['memory_capacity']}",
        "",
        "# 探索-利用平衡",
        "# exploration_rate: 探索比例 (0.0=纯利用, 1.0=纯探索)",
        f"exploration_rate: {config['exploration_rate']}",
        "# epsilon: ε-贪心策略的随机探索概率",
        f"epsilon: {config['epsilon']}",
        "# interrupt_threshold: 中断阈值（预测置信度低于此值时触发中断）",
        f"interrupt_threshold: {config['interrupt_threshold']}",
        "",
        "# 预测器参数",
        f"feature_dimensions: {config['feature_dimensions']}",
        f"low_alpha_init: {config['low_alpha_init']}",
        f"low_alpha_min: {config['low_alpha_min']}",
        f"low_alpha_max: {config['low_alpha_max']}",
        f"low_adaptation_rate: {config['low_adaptation_rate']}",
        f"mid_window_size: {config['mid_window_size']}",
        f"mid_forecast_horizon: {config['mid_forecast_horizon']}",
        f"high_window_size: {config['high_window_size']}",
        f"high_reversion_strength: {config['high_reversion_strength']}",
        f"ensemble_beta: {config['ensemble_beta']}",
        "",
        "# 监控",
        f"cpu_interval_seconds: {config['cpu_interval_seconds']}",
        "disk_paths:",
        "  - \"/\"",
        "log_paths: []",
        f"max_log_lines: {config['max_log_lines']}",
        "",
        "# 驱力基础权重（safety + gain + efficiency = 1.0）",
        "base_weights:",
        f"  safety: {config['base_weights']['safety']}",
        f"  gain: {config['base_weights']['gain']}",
        f"  efficiency: {config['base_weights']['efficiency']}",
        "",
        "# 分页查询",
        f"query_page_size: {config['query_page_size']}",
        f"batch_query_limit: {config['batch_query_limit']}",
        "",
    ]
    return "\n".join(lines)


async def _cmd_doctor(args: argparse.Namespace) -> int:
    """环境诊断：检查Python版本、依赖、配置、磁盘、端口。

    第一性原理：主动发现问题而非等用户报错。
    用户运行报错时往往不知根因，doctor命令系统性排查常见问题，
    将"被动排错"转为"主动诊断"。
    """
    print(_colorize("=== Self-Evolving Agent Doctor ===", _COLOR_BOLD))
    print()

    checks_passed = 0
    checks_failed = 0
    checks_warning = 0

    def _ok(name: str, detail: str = "") -> None:
        nonlocal checks_passed
        checks_passed += 1
        msg = f"  {_colorize('[OK]', _COLOR_GREEN)} {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)

    def _fail(name: str, detail: str, fix: str = "") -> None:
        nonlocal checks_failed
        checks_failed += 1
        print(f"  {_colorize('[FAIL]', _COLOR_RED)} {name} — {detail}")
        if fix:
            print(f"          {_colorize('Fix:', _COLOR_YELLOW)} {fix}")

    def _warn(name: str, detail: str, tip: str = "") -> None:
        nonlocal checks_warning
        checks_warning += 1
        print(f"  {_colorize('[WARN]', _COLOR_YELLOW)} {name} — {detail}")
        if tip:
            print(f"          {_colorize('Tip:', _COLOR_CYAN)} {tip}")

    # 1. Python版本检查
    print(_colorize("[1/6] Python Runtime", _COLOR_BOLD))
    py_version = sys.version_info
    py_str = f"{py_version.major}.{py_version.minor}.{py_version.micro}"
    if py_version >= (3, 9):
        _ok("Python version", py_str)
    else:
        _fail("Python version", f"{py_str} (需要 >=3.9)", "升级Python到3.9+")

    # 2. 依赖检查
    print()
    print(_colorize("[2/6] Dependencies", _COLOR_BOLD))
    required_deps = [
        ("numpy", "numpy"),
        ("yaml", "pyyaml"),
        ("psutil", "psutil"),
    ]
    for import_name, pkg_name in required_deps:
        try:
            mod = __import__(import_name)
            version = getattr(mod, "__version__", "unknown")
            _ok(f"{pkg_name}", version)
        except ImportError:
            _fail(
                f"{pkg_name}", "not installed",
                f"pip install {pkg_name}",
            )

    # 3. 核心模块导入检查
    print()
    print(_colorize("[3/6] Core Modules", _COLOR_BOLD))
    core_modules = [
        "agent_core",
        "agent_core.cli",
        "agent_core.common.logger",
        "agent_core.common.config_validator",
        "agent_core.common.environment_config",
    ]
    for mod_name in core_modules:
        try:
            __import__(mod_name)
            _ok(mod_name)
        except ImportError as exc:
            _fail(mod_name, str(exc), "检查安装: pip install -e .")
        except Exception as exc:
            _warn(mod_name, f"import error: {exc}")

    # 4. 配置文件检查
    print()
    print(_colorize("[4/6] Configuration", _COLOR_BOLD))
    config_dir = os.environ.get(
        "AGENT_CONFIG_DIR",
        os.path.join(os.getcwd(), "config"),
    )
    if os.path.isdir(config_dir):
        _ok("Config directory", config_dir)
        # 检查base.yaml
        base_path = os.path.join(config_dir, "base.yaml")
        if os.path.exists(base_path):
            _ok("base.yaml", "found")
        else:
            _warn("base.yaml", "not found", "运行 'agent init' 生成配置")
        # 检查环境配置
        env_path = os.path.join(config_dir, f"{args.env}.yaml")
        if os.path.exists(env_path):
            _ok(f"{args.env}.yaml", "found")
        else:
            _warn(f"{args.env}.yaml", "not found", f"运行 'agent init --env {args.env}'")
    else:
        _warn("Config directory", f"{config_dir} not found", "运行 'agent init' 创建配置")

    # 5. 数据目录权限检查
    print()
    print(_colorize("[5/6] Data Directories", _COLOR_BOLD))
    data_dirs = [
        ("data", os.path.join(os.getcwd(), "data")),
        ("data/logs", os.path.join(os.getcwd(), "data", "logs")),
    ]
    for name, path in data_dirs:
        if os.path.isdir(path):
            # 检查可写
            if os.access(path, os.W_OK):
                _ok(name, f"writable ({path})")
            else:
                _fail(name, "not writable", f"chmod +w {path}")
        else:
            # 尝试创建
            try:
                os.makedirs(path, exist_ok=True)
                _ok(name, f"created ({path})")
            except OSError as exc:
                _warn(name, f"cannot create: {exc}", f"mkdir -p {path}")

    # 6. 端口可用性检查
    print()
    print(_colorize("[6/6] Network Ports", _COLOR_BOLD))
    import socket
    ports_to_check = [
        (args.dashboard_port, "Dashboard"),
    ]
    for port, service in ports_to_check:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        try:
            result = sock.connect_ex(("127.0.0.1", port))
            if result == 0:
                _warn(
                    f"Port {port} ({service})",
                    "in use",
                    f"Agent可能已在运行，或用 --port 指定其他端口",
                )
            else:
                _ok(f"Port {port} ({service})", "available")
        finally:
            sock.close()

    # 汇总
    print()
    print(_colorize("=== Summary ===", _COLOR_BOLD))
    total = checks_passed + checks_failed + checks_warning
    print(f"  Passed:  {checks_passed}/{total}")
    print(f"  Failed:  {checks_failed}/{total}")
    print(f"  Warning: {checks_warning}/{total}")

    if checks_failed > 0:
        _print_error(f"{checks_failed} check(s) failed. Fix them before starting the Agent.")
        return 1
    elif checks_warning > 0:
        _print_warning(f"{checks_warning} warning(s). Agent may still run, but check them.")
        return 0
    else:
        _print_success("All checks passed. Agent is ready to run!")
        return 0


async def _cmd_completion(args: argparse.Namespace) -> int:
    """生成Shell补全脚本。

    第一性原理：Tab补全减少记忆负担——用户不需要记住所有子命令和参数，
    按Tab即可发现可用选项。降低"使用阻力"。
    """
    shell = args.shell

    scripts = {
        "bash": _generate_bash_completion(),
        "zsh": _generate_zsh_completion(),
        "fish": _generate_fish_completion(),
    }

    if shell not in scripts:
        _print_error(f"Unsupported shell: {shell}")
        _print_info(f"Supported shells: {', '.join(scripts.keys())}")
        return 1

    script = scripts[shell]

    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(script)
            _print_success(f"Completion script written to {args.output}")
            _print_install_hint(shell, args.output)
        except OSError as exc:
            _print_error(f"Failed to write: {exc}")
            return 1
    else:
        print(script)

    return 0


def _print_install_hint(shell: str, output_path: str) -> None:
    """打印安装提示。"""
    hints = {
        "bash": [
            f"  source {output_path}",
            f"  # 或添加到 ~/.bashrc 实现持久化:",
            f"  echo 'source {output_path}' >> ~/.bashrc",
        ],
        "zsh": [
            f"  source {output_path}",
            f"  # 或添加到 ~/.zshrc 实现持久化:",
            f"  echo 'source {output_path}' >> ~/.zshrc",
        ],
        "fish": [
            f"  # fish自动加载 ~/.config/fish/completions/ 下的脚本",
            f"  cp {output_path} ~/.config/fish/completions/agent.fish",
        ],
    }
    _print_info("To enable completion:")
    for line in hints.get(shell, []):
        print(line)


def _generate_bash_completion() -> str:
    """生成bash补全脚本。"""
    return """# bash completion for Self-Evolving Agent
# Generated by 'agent completion --shell bash'

_agent_completions() {
    local cur prev words cword
    _init_completion || return

    local commands="start stop status health dashboard version log init doctor completion"
    local global_opts="--help --version -h -V"

    # 第一个参数：补全子命令
    if [ $cword -eq 1 ]; then
        COMPREPLY=( $(compgen -W "${commands} ${global_opts}" -- "${cur}") )
        return 0
    fi

    # 子命令的参数补全
    local cmd="${words[1]}"
    case "${cmd}" in
        start)
            local opts="--env --config --log-level --online-interval --offline-interval --dry-run --help -h"
            COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
            # --env 的值补全
            if [[ "${prev}" == "--env" ]]; then
                COMPREPLY=( $(compgen -W "dev prod test" -- "${cur}") )
            fi
            # --log-level 的值补全
            if [[ "${prev}" == "--log-level" ]]; then
                COMPREPLY=( $(compgen -W "DEBUG INFO WARNING ERROR CRITICAL" -- "${cur}") )
            fi
            # --config 的文件补全
            if [[ "${prev}" == "--config" ]]; then
                COMPREPLY=( $(compgen -f -X "*.yaml" -- "${cur}") )
            fi
            ;;
        stop|status|dashboard)
            local opts="--port --help -h"
            COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
            ;;
        health)
            local opts="--env --config --help -h"
            COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
            if [[ "${prev}" == "--env" ]]; then
                COMPREPLY=( $(compgen -W "dev prod test" -- "${cur}") )
            fi
            ;;
        log)
            local opts="--lines --help -h"
            COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
            ;;
        init)
            local opts="--env --output --non-interactive --force --log-level --online-interval --offline-interval --memory-capacity --exploration-rate --help -h"
            COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
            if [[ "${prev}" == "--env" ]]; then
                COMPREPLY=( $(compgen -W "dev prod test" -- "${cur}") )
            fi
            ;;
        doctor)
            local opts="--env --dashboard-port --help -h"
            COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
            ;;
        completion)
            local opts="--shell --output --help -h"
            COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
            if [[ "${prev}" == "--shell" ]]; then
                COMPREPLY=( $(compgen -W "bash zsh fish" -- "${cur}") )
            fi
            ;;
        *)
            COMPREPLY=( $(compgen -W "${global_opts}" -- "${cur}") )
            ;;
    esac

    return 0
}

complete -F _agent_completions agent self-evolving-agent
"""


def _generate_zsh_completion() -> str:
    """生成zsh补全脚本。"""
    return """#compdef agent self-evolving-agent
# zsh completion for Self-Evolving Agent
# Generated by 'agent completion --shell zsh'

_agent() {
    local context state line
    typeset -A opt_args

    _arguments -C \\
        "1:command:(start stop status health dashboard version log init doctor completion)" \\
        "*::arg:->args"

    case $state in
        args)
            case ${line[1]} in
                start)
                    _arguments \\
                        "--env[环境]:env:(dev prod test)" \\
                        "--config[配置文件]:file:_files -g '*.yaml'" \\
                        "--log-level[日志级别]:level:(DEBUG INFO WARNING ERROR CRITICAL)" \\
                        "--online-interval[在线间隔]:float:" \\
                        "--offline-interval[离线间隔]:float:" \\
                        "--dry-run[仅验证配置]"
                    ;;
                stop|status|dashboard)
                    _arguments "--port[端口]:port:"
                    ;;
                health)
                    _arguments \\
                        "--env[环境]:env:(dev prod test)" \\
                        "--config[配置文件]:file:_files -g '*.yaml'"
                    ;;
                log)
                    _arguments "--lines[行数]:lines:"
                    ;;
                init)
                    _arguments \\
                        "--env[环境]:env:(dev prod test)" \\
                        "--output[输出路径]:file:_files" \\
                        "--non-interactive[非交互模式]" \\
                        "--force[覆盖已存在]" \\
                        "--log-level[日志级别]:level:(DEBUG INFO WARNING ERROR CRITICAL)" \\
                        "--online-interval[在线间隔]:float:" \\
                        "--offline-interval[离线间隔]:float:" \\
                        "--memory-capacity[记忆容量]:int:" \\
                        "--exploration-rate[探索率]:float:"
                    ;;
                doctor)
                    _arguments \\
                        "--env[环境]:env:(dev prod test)" \\
                        "--dashboard-port[端口]:port:"
                    ;;
                completion)
                    _arguments \\
                        "--shell[shell类型]:shell:(bash zsh fish)" \\
                        "--output[输出路径]:file:_files"
                    ;;
            esac
            ;;
    esac
}

_agent "$@"
"""


def _generate_fish_completion() -> str:
    """生成fish补全脚本。"""
    return """# fish completion for Self-Evolving Agent
# Generated by 'agent completion --shell fish'

# 子命令补全
complete -c agent -f -n "__fish_use_subcommand" -a "start" -d "启动Agent"
complete -c agent -f -n "__fish_use_subcommand" -a "stop" -d "停止Agent"
complete -c agent -f -n "__fish_use_subcommand" -a "status" -d "查询状态"
complete -c agent -f -n "__fish_use_subcommand" -a "health" -d "健康检查"
complete -c agent -f -n "__fish_use_subcommand" -a "dashboard" -d "启动Dashboard"
complete -c agent -f -n "__fish_use_subcommand" -a "version" -d "显示版本"
complete -c agent -f -n "__fish_use_subcommand" -a "log" -d "查看日志"
complete -c agent -f -n "__fish_use_subcommand" -a "init" -d "生成配置"
complete -c agent -f -n "__fish_use_subcommand" -a "doctor" -d "环境诊断"
complete -c agent -f -n "__fish_use_subcommand" -a "completion" -d "生成补全脚本"

# start 参数
complete -c agent -f -n "__fish_seen_subcommand_from start" -l env -d "环境" -a "dev prod test"
complete -c agent -f -n "__fish_seen_subcommand_from start" -l config -d "配置文件" -r
complete -c agent -f -n "__fish_seen_subcommand_from start" -l log-level -d "日志级别" -a "DEBUG INFO WARNING ERROR CRITICAL"
complete -c agent -f -n "__fish_seen_subcommand_from start" -l online-interval -d "在线间隔" -r
complete -c agent -f -n "__fish_seen_subcommand_from start" -l offline-interval -d "离线间隔" -r
complete -c agent -f -n "__fish_seen_subcommand_from start" -l dry-run -d "仅验证配置"

# stop/status/dashboard 参数
complete -c agent -f -n "__fish_seen_subcommand_from stop status dashboard" -l port -d "端口" -r

# health 参数
complete -c agent -f -n "__fish_seen_subcommand_from health" -l env -d "环境" -a "dev prod test"
complete -c agent -f -n "__fish_seen_subcommand_from health" -l config -d "配置文件" -r

# log 参数
complete -c agent -f -n "__fish_seen_subcommand_from log" -l lines -d "行数" -r

# init 参数
complete -c agent -f -n "__fish_seen_subcommand_from init" -l env -d "环境" -a "dev prod test"
complete -c agent -f -n "__fish_seen_subcommand_from init" -l output -d "输出路径" -r
complete -c agent -f -n "__fish_seen_subcommand_from init" -l non-interactive -d "非交互模式"
complete -c agent -f -n "__fish_seen_subcommand_from init" -l force -d "覆盖已存在"
complete -c agent -f -n "__fish_seen_subcommand_from init" -l log-level -d "日志级别" -a "DEBUG INFO WARNING ERROR CRITICAL"
complete -c agent -f -n "__fish_seen_subcommand_from init" -l online-interval -d "在线间隔" -r
complete -c agent -f -n "__fish_seen_subcommand_from init" -l offline-interval -d "离线间隔" -r
complete -c agent -f -n "__fish_seen_subcommand_from init" -l memory-capacity -d "记忆容量" -r
complete -c agent -f -n "__fish_seen_subcommand_from init" -l exploration-rate -d "探索率" -r

# doctor 参数
complete -c agent -f -n "__fish_seen_subcommand_from doctor" -l env -d "环境" -a "dev prod test"
complete -c agent -f -n "__fish_seen_subcommand_from doctor" -l dashboard-port -d "端口" -r

# completion 参数
complete -c agent -f -n "__fish_seen_subcommand_from completion" -l shell -d "shell类型" -a "bash zsh fish"
complete -c agent -f -n "__fish_seen_subcommand_from completion" -l output -d "输出路径" -r
"""


async def _cmd_config(args: argparse.Namespace) -> int:
    """配置管理：show（查看生效配置）和 template（生成全量模板）。

    第一性原理：降低进阶阻力——提供可定制的起点。
    - show: 展示当前生效配置及来源，帮助用户理解"配置从哪来"
    - template: 生成全量带注释模板，供进阶用户参考定制
    """
    action = args.config_action

    if action == "show":
        return _config_show(args)
    elif action == "template":
        return _config_template(args)
    else:
        _print_error(f"Unknown config action: {action}")
        return 1


def _config_show(args: argparse.Namespace) -> int:
    """显示当前生效配置及来源。"""
    print(_colorize("=== Effective Configuration ===", _COLOR_BOLD))
    print()

    # 显示配置来源
    config_dir = os.environ.get(
        "AGENT_CONFIG_DIR",
        os.path.join(os.getcwd(), "config"),
    )
    print(_colorize("Configuration Sources:", _COLOR_BOLD))

    sources = []
    if args.config:
        sources.append(("CLI --config", args.config, os.path.exists(args.config)))
    else:
        base_path = os.path.join(config_dir, "base.yaml")
        sources.append(("base.yaml", base_path, os.path.exists(base_path)))
        env_path = os.path.join(config_dir, f"{args.env}.yaml")
        sources.append((f"{args.env}.yaml", env_path, os.path.exists(env_path)))
        sources.append(("built-in defaults", "(internal)", True))

    for name, path, exists in sources:
        status = _colorize("[OK]", _COLOR_GREEN) if exists else _colorize("[MISSING]", _COLOR_RED)
        print(f"  {status} {name:<20} {path}")

    print()

    # 加载并显示生效配置
    config = _load_config(args.env, args.config)
    print(_colorize("Effective Values:", _COLOR_BOLD))

    # 按类别分组显示
    categories = [
        ("Basic", ["app_name", "app_version", "log_level"]),
        ("Loop", ["online_interval_seconds", "offline_interval_seconds"]),
        ("Memory", ["memory_capacity"]),
        ("Exploration", ["exploration_rate", "epsilon", "interrupt_threshold"]),
        ("Predictor", [
            "feature_dimensions", "low_alpha_init", "low_alpha_min",
            "low_alpha_max", "low_adaptation_rate", "mid_window_size",
            "mid_forecast_horizon", "high_window_size",
            "high_reversion_strength", "ensemble_beta",
        ]),
        ("Monitor", ["cpu_interval_seconds", "max_log_lines"]),
        ("Drives", ["base_weights"]),
        ("Query", ["query_page_size", "batch_query_limit"]),
    ]

    for cat_name, keys in categories:
        print(f"  {_colorize(f'[{cat_name}]', _COLOR_CYAN)}")
        for key in keys:
            if key in config:
                val = config[key]
                if isinstance(val, dict):
                    print(f"    {key:<30} {val}")
                else:
                    print(f"    {key:<30} {val}")

    print()
    _print_info("Use 'agent config template' to see all options with descriptions.")
    return 0


def _config_template(args: argparse.Namespace) -> int:
    """生成全量带注释配置模板。"""
    template = _generate_full_template()

    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(template)
            _print_success(f"Template written to {args.output}")
            _print_info("Edit the file, then: agent start --config <file> --dry-run")
        except OSError as exc:
            _print_error(f"Failed to write: {exc}")
            return 1
    else:
        print(template)

    return 0


def _generate_full_template() -> str:
    """生成全量带注释配置模板。

    与 init 生成的精简配置不同，template 包含所有可配置项的
    详细说明、取值范围和示例，供进阶用户参考。
    """
    return """# Self-Evolving Agent — 全量配置模板
# 由 'agent config template' 生成
# 
# 使用方法:
#   1. 复制此文件: agent config template --output my_config.yaml
#   2. 编辑 my_config.yaml，按需调整参数
#   3. 验证: agent start --config my_config.yaml --dry-run
#   4. 启动: agent start --config my_config.yaml

# ============================================================
# 基础配置
# ============================================================
app_name: "self_evolving_agent"        # 应用名（用于日志和状态标识）
app_version: "1.0.0"                   # 应用版本

# 日志级别: DEBUG | INFO | WARNING | ERROR | CRITICAL
# DEBUG: 详细调试信息（开发环境）
# INFO: 一般运行信息（默认）
# WARNING: 警告（不影响运行）
# ERROR: 错误（影响部分功能）
# CRITICAL: 严重错误（影响整体运行）
log_level: "INFO"

# ============================================================
# 运行循环配置
# ============================================================
# 在线循环：感知-决策-行动周期
# 值越小响应越快，CPU占用越高
online_interval_seconds: 1.0           # 范围: 0.1-60.0

# 离线循环：经验回放与模型压缩
# 值越小优化越频繁，资源消耗越高
offline_interval_seconds: 3600.0       # 范围: 60.0-86400.0

# ============================================================
# 记忆系统
# ============================================================
# 存储的经验轨迹数量上限
# 值越大记忆越长，内存占用越高
memory_capacity: 10000                 # 范围: 100-1000000

# ============================================================
# 探索-利用平衡
# ============================================================
# exploration_rate: 探索比例
#   0.0 = 纯利用（总是选择最优）
#   1.0 = 纯探索（随机选择）
#   推荐: 0.2-0.4（初期偏高，后期降低）
exploration_rate: 0.3                  # 范围: 0.0-1.0

# epsilon: ε-贪心策略的随机探索概率
# 以 epsilon 概率随机探索，1-epsilon 概率利用最优
epsilon: 0.2                           # 范围: 0.0-1.0

# interrupt_threshold: 中断阈值
# 预测置信度低于此值时触发中断，重新决策
# 值越高越敏感（更容易中断）
interrupt_threshold: 0.7               # 范围: 0.0-1.0

# ============================================================
# 预测器参数
# ============================================================
# 特征维度：输入特征的维度数
feature_dimensions: 10                 # 范围: 1-100

# 低频预测器（长期趋势）
low_alpha_init: 0.3                    # 初始学习率
low_alpha_min: 0.05                    # 最小学习率
low_alpha_max: 0.8                     # 最大学习率
low_adaptation_rate: 0.1               # 学习率自适应速率

# 中频预测器（中期模式）
mid_window_size: 20                    # 滑动窗口大小
mid_forecast_horizon: 5                # 预测视野

# 高频预测器（短期波动）
high_window_size: 100                  # 均值回归窗口
high_reversion_strength: 0.3           # 均值回归强度

# 集成预测器
# beta: 集成权重温度参数，值越高越偏向最优预测器
ensemble_beta: 5.0                     # 范围: 1.0-20.0

# ============================================================
# 监控配置
# ============================================================
# CPU监控采样间隔
cpu_interval_seconds: 1.0              # 范围: 0.5-60.0

# 磁盘监控路径
disk_paths:
  - "/"

# 日志文件监控路径（空列表=不监控）
log_paths: []

# 日志最大保留行数
max_log_lines: 100                     # 范围: 10-10000

# ============================================================
# 驱力系统
# ============================================================
# 基础权重（三者之和必须为1.0）
# safety: 安全驱力（避免危险）
# gain: 收益驱力（追求目标）
# efficiency: 效率驱力（资源优化）
base_weights:
  safety: 0.4
  gain: 0.4
  efficiency: 0.2

# ============================================================
# 查询配置
# ============================================================
# 分页查询每页大小
query_page_size: 20                    # 范围: 1-1000

# 批量查询上限
batch_query_limit: 1000                # 范围: 10-100000
"""


def _build_parser() -> argparse.ArgumentParser:
    """构建CLI参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="agent",
        description="Self-Evolving Agent - 自驱永续进化智能体",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  agent init                     # 交互式生成配置文件
  agent init --non-interactive   # 非交互模式生成默认配置
  agent doctor                   # 环境诊断
  agent start                    # 使用默认配置启动
  agent start --env prod         # 使用生产环境配置启动
  agent start --config my.yaml   # 使用指定配置启动
  agent start --dry-run          # 仅验证配置不启动
  agent stop                     # 停止运行中的Agent
  agent status                   # 查询运行状态
  agent health                   # 健康检查
  agent log --lines 50           # 查看最近50行日志
  agent dashboard --port 9090    # 启动Dashboard
  agent version                  # 显示版本
""",
    )

    # --version / -V 全局标志
    parser.add_argument(
        "--version", "-V", action="store_true",
        help="显示版本信息并退出",
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # start 子命令
    start_parser = subparsers.add_parser(
        "start", help="启动Agent"
    )
    start_parser.add_argument(
        "--env", default=_DEFAULT_ENV,
        help="环境 (dev/prod/test, 默认: dev)"
    )
    start_parser.add_argument(
        "--config", default=None,
        help="配置文件路径 (覆盖--env)"
    )
    start_parser.add_argument(
        "--log-level", default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="日志级别"
    )
    start_parser.add_argument(
        "--online-interval", type=float, default=None,
        help="在线循环间隔(秒)"
    )
    start_parser.add_argument(
        "--offline-interval", type=float, default=None,
        help="离线循环间隔(秒)"
    )
    start_parser.add_argument(
        "--dry-run", action="store_true",
        help="仅验证配置不启动Agent"
    )

    # stop 子命令
    stop_parser = subparsers.add_parser(
        "stop", help="停止运行中的Agent"
    )
    stop_parser.add_argument(
        "--port", type=int, default=_DEFAULT_DASHBOARD_PORT,
        help=f"Dashboard端口 (默认: {_DEFAULT_DASHBOARD_PORT})"
    )

    # status 子命令
    status_parser = subparsers.add_parser(
        "status", help="查询Agent运行状态"
    )
    status_parser.add_argument(
        "--port", type=int, default=_DEFAULT_DASHBOARD_PORT,
        help=f"Dashboard端口 (默认: {_DEFAULT_DASHBOARD_PORT})"
    )

    # health 子命令
    health_parser = subparsers.add_parser(
        "health", help="健康检查"
    )
    health_parser.add_argument(
        "--env", default=_DEFAULT_ENV,
        help="环境 (默认: dev)"
    )
    health_parser.add_argument(
        "--config", default=None,
        help="配置文件路径"
    )

    # dashboard 子命令
    dash_parser = subparsers.add_parser(
        "dashboard", help="启动独立Dashboard服务"
    )
    dash_parser.add_argument(
        "--port", type=int, default=_DEFAULT_DASHBOARD_PORT,
        help=f"端口 (默认: {_DEFAULT_DASHBOARD_PORT})"
    )

    # version 子命令
    subparsers.add_parser(
        "version", help="显示版本信息"
    )

    # log 子命令
    log_parser = subparsers.add_parser(
        "log", help="查看最近日志"
    )
    log_parser.add_argument(
        "--lines", type=int, default=20,
        help="显示行数 (默认: 20)"
    )

    # init 子命令
    init_parser = subparsers.add_parser(
        "init", help="交互式生成配置文件"
    )
    init_parser.add_argument(
        "--env", default=_DEFAULT_ENV,
        help="环境名，决定输出文件名 (dev/prod/test, 默认: dev)"
    )
    init_parser.add_argument(
        "--output", default=None,
        help="输出文件路径 (覆盖默认 config/{env}.yaml)"
    )
    init_parser.add_argument(
        "--non-interactive", action="store_true",
        help="非交互模式，使用默认值或命令行参数"
    )
    init_parser.add_argument(
        "--force", action="store_true",
        help="覆盖已存在的配置文件"
    )
    init_parser.add_argument(
        "--log-level", default=None,
        help="日志级别 (非交互模式使用)"
    )
    init_parser.add_argument(
        "--online-interval", type=float, default=None,
        help="在线循环间隔秒数 (非交互模式使用)"
    )
    init_parser.add_argument(
        "--offline-interval", type=float, default=None,
        help="离线循环间隔秒数 (非交互模式使用)"
    )
    init_parser.add_argument(
        "--memory-capacity", type=int, default=None,
        help="记忆容量 (非交互模式使用)"
    )
    init_parser.add_argument(
        "--exploration-rate", type=float, default=None,
        help="探索率 0.0-1.0 (非交互模式使用)"
    )

    # doctor 子命令
    doctor_parser = subparsers.add_parser(
        "doctor", help="环境诊断 — 检查Python/依赖/配置/端口"
    )
    doctor_parser.add_argument(
        "--env", default=_DEFAULT_ENV,
        help="检查指定环境的配置 (默认: dev)"
    )
    doctor_parser.add_argument(
        "--dashboard-port", type=int, default=_DEFAULT_DASHBOARD_PORT,
        help=f"检查Dashboard端口可用性 (默认: {_DEFAULT_DASHBOARD_PORT})"
    )

    # completion 子命令
    completion_parser = subparsers.add_parser(
        "completion", help="生成Shell补全脚本"
    )
    completion_parser.add_argument(
        "--shell", required=True,
        choices=["bash", "zsh", "fish"],
        help="目标shell类型"
    )
    completion_parser.add_argument(
        "--output", default=None,
        help="输出文件路径 (不指定则输出到stdout)"
    )

    # config 子命令组
    config_parser = subparsers.add_parser(
        "config", help="配置管理 — show/template"
    )
    config_sub = config_parser.add_subparsers(
        dest="config_action", help="配置操作"
    )
    # config show
    config_show = config_sub.add_parser(
        "show", help="显示当前生效配置及来源"
    )
    config_show.add_argument(
        "--env", default=_DEFAULT_ENV,
        help="环境 (默认: dev)"
    )
    config_show.add_argument(
        "--config", default=None,
        help="配置文件路径"
    )
    # config template
    config_template = config_sub.add_parser(
        "template", help="生成全量带注释配置模板"
    )
    config_template.add_argument(
        "--output", default=None,
        help="输出文件路径 (不指定则输出到stdout)"
    )

    return parser


def _print_interactive_help() -> None:
    """打印交互式帮助：推荐命令和常见工作流。

    第一性原理：降低发现阻力——用户输入 'agent' 无参数时，
    不应只显示干瘪的usage，而应主动展示推荐命令和常见工作流，
    引导用户从"不知道做什么"到"知道下一步"。
    """
    print(_colorize("Self-Evolving Agent", _COLOR_BOLD) + " — 自驱永续进化智能体")
    print()
    print(_colorize("Quick Start:", _COLOR_BOLD))
    print(f"  {_colorize('agent doctor', _COLOR_CYAN)}                    # 1. 环境诊断")
    print(f"  {_colorize('agent init', _COLOR_CYAN)}                     # 2. 生成配置")
    print(f"  {_colorize('agent start --dry-run', _COLOR_CYAN)}          # 3. 验证配置")
    print(f"  {_colorize('agent start', _COLOR_CYAN)}                    # 4. 启动Agent")
    print()

    print(_colorize("Commands:", _COLOR_BOLD))
    commands = [
        ("init", "交互式生成配置文件"),
        ("doctor", "环境诊断 — 检查Python/依赖/配置/端口"),
        ("config", "配置管理 — show/template"),
        ("start", "启动Agent"),
        ("stop", "停止运行中的Agent"),
        ("status", "查询Agent运行状态"),
        ("health", "健康检查"),
        ("log", "查看最近日志"),
        ("dashboard", "启动独立Dashboard服务"),
        ("completion", "生成Shell补全脚本"),
        ("version", "显示版本信息"),
    ]
    for cmd, desc in commands:
        print(f"  {_colorize(f'agent {cmd:<12}', _COLOR_CYAN)}  {desc}")

    print()
    print(_colorize("Common Workflows:", _COLOR_BOLD))
    print(f"  {_colorize('首次使用:', _COLOR_YELLOW)}")
    print(f"    agent doctor && agent init && agent start")
    print(f"  {_colorize('查看状态:', _COLOR_YELLOW)}")
    print(f"    agent status && agent log --lines 50")
    print(f"  {_colorize('生产部署:', _COLOR_YELLOW)}")
    print(f"    agent init --env prod --non-interactive && agent start --env prod")
    print(f"  {_colorize('进阶配置:', _COLOR_YELLOW)}")
    print(f"    agent config template && agent config show")
    print()

    print(_colorize("Tips:", _COLOR_BOLD))
    print(f"  • 使用 {_colorize('--help', _COLOR_CYAN)} 查看任何命令的详细选项")
    print(f"  • 使用 {_colorize('agent completion --shell bash', _COLOR_CYAN)} 启用Tab补全")
    print(f"  • 文档: {_colorize('docs/user/quick_start.md', _COLOR_CYAN)}")
    print()


def main(argv: Optional[List[str]] = None) -> int:
    """
    CLI主入口。

    Args:
        argv: 命令行参数（None时使用sys.argv）。

    Returns:
        退出码（0成功，非0失败）。
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # --version 全局标志
    if args.version:
        return asyncio.run(_cmd_version(args))

    if args.command is None:
        _print_interactive_help()
        return 0

    # 分发到子命令
    handlers = {
        "start": _cmd_start,
        "stop": _cmd_stop,
        "status": _cmd_status,
        "health": _cmd_health,
        "dashboard": _cmd_dashboard,
        "version": _cmd_version,
        "log": _cmd_log,
        "init": _cmd_init,
        "doctor": _cmd_doctor,
        "completion": _cmd_completion,
        "config": _cmd_config,
    }

    handler = handlers.get(args.command)
    if handler is None:
        _print_error(f"Unknown command: {args.command}")
        return 1

    try:
        return asyncio.run(handler(args))
    except KeyboardInterrupt:
        _print_info("\nInterrupted")
        return 130
    except Exception as exc:
        _print_error(f"Unexpected error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

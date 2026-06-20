"""
U-1: CLI统一入口与可启动性修复 测试。

覆盖：CLI参数解析、子命令分发、配置加载、默认值、输出格式。
"""

import asyncio
import io
import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from agent_core.cli import (
    main,
    _build_parser,
    _load_config,
    _get_default_config,
    _colorize,
    _print_success,
    _print_error,
    _print_warning,
    _print_info,
)


class TestCLIParser(unittest.TestCase):
    """CLI参数解析测试。"""

    def test_no_command_shows_help(self):
        """无命令显示帮助。"""
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            result = main([])
        output = mock_stdout.getvalue()
        self.assertIn("Self-Evolving Agent", output)
        self.assertEqual(result, 0)

    def test_start_command_parsed(self):
        """start命令解析。"""
        parser = _build_parser()
        args = parser.parse_args(["start"])
        self.assertEqual(args.command, "start")
        self.assertEqual(args.env, "dev")

    def test_start_with_env(self):
        """start带环境参数。"""
        parser = _build_parser()
        args = parser.parse_args(["start", "--env", "prod"])
        self.assertEqual(args.env, "prod")

    def test_start_with_config(self):
        """start带配置文件。"""
        parser = _build_parser()
        args = parser.parse_args(["start", "--config", "/tmp/my.yaml"])
        self.assertEqual(args.config, "/tmp/my.yaml")

    def test_start_with_log_level(self):
        """start带日志级别。"""
        parser = _build_parser()
        args = parser.parse_args(["start", "--log-level", "DEBUG"])
        self.assertEqual(args.log_level, "DEBUG")

    def test_start_with_intervals(self):
        """start带间隔参数。"""
        parser = _build_parser()
        args = parser.parse_args([
            "start", "--online-interval", "2.0", "--offline-interval", "7200"
        ])
        self.assertEqual(args.online_interval, 2.0)
        self.assertEqual(args.offline_interval, 7200.0)

    def test_status_command_parsed(self):
        """status命令解析。"""
        parser = _build_parser()
        args = parser.parse_args(["status"])
        self.assertEqual(args.command, "status")
        self.assertEqual(args.port, 8080)

    def test_status_with_port(self):
        """status带端口。"""
        parser = _build_parser()
        args = parser.parse_args(["status", "--port", "9090"])
        self.assertEqual(args.port, 9090)

    def test_health_command_parsed(self):
        """health命令解析。"""
        parser = _build_parser()
        args = parser.parse_args(["health"])
        self.assertEqual(args.command, "health")

    def test_dashboard_command_parsed(self):
        """dashboard命令解析。"""
        parser = _build_parser()
        args = parser.parse_args(["dashboard", "--port", "9090"])
        self.assertEqual(args.command, "dashboard")
        self.assertEqual(args.port, 9090)

    def test_version_command_parsed(self):
        """version命令解析。"""
        parser = _build_parser()
        args = parser.parse_args(["version"])
        self.assertEqual(args.command, "version")

    def test_invalid_log_level_rejected(self):
        """无效日志级别被拒绝。"""
        parser = _build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["start", "--log-level", "INVALID"])


class TestConfigLoading(unittest.TestCase):
    """配置加载测试。"""

    def test_default_config_has_required_keys(self):
        """默认配置包含必需键。"""
        config = _get_default_config()
        required_keys = [
            "app_name", "app_version", "log_level",
            "online_interval_seconds", "offline_interval_seconds",
            "memory_capacity", "exploration_rate", "epsilon",
            "feature_dimensions", "base_weights",
        ]
        for key in required_keys:
            self.assertIn(key, config, f"Missing key: {key}")

    def test_default_config_weights_sum_to_one(self):
        """默认权重和为1。"""
        config = _get_default_config()
        weights = config["base_weights"]
        total = weights["safety"] + weights["gain"] + weights["efficiency"]
        self.assertAlmostEqual(total, 1.0, places=2)

    def test_load_config_no_file_uses_defaults(self):
        """无配置文件时使用默认值。"""
        config = _load_config("dev", None)
        self.assertEqual(config["app_name"], "self_evolving_agent")
        self.assertGreater(config["online_interval_seconds"], 0)

    def test_load_config_from_file(self):
        """从文件加载配置。"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("app_name: test_agent\nonline_interval_seconds: 5.0\n")
            f.flush()
            try:
                config = _load_config("dev", f.name)
                self.assertEqual(config["app_name"], "test_agent")
                self.assertEqual(config["online_interval_seconds"], 5.0)
                # 未指定的键使用默认值
                self.assertIn("memory_capacity", config)
            finally:
                os.unlink(f.name)

    def test_load_config_file_overrides_defaults(self):
        """配置文件覆盖默认值。"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("memory_capacity: 99999\n")
            f.flush()
            try:
                config = _load_config("dev", f.name)
                self.assertEqual(config["memory_capacity"], 99999)
            finally:
                os.unlink(f.name)

    def test_load_config_nonexistent_file_uses_defaults(self):
        """不存在的配置文件使用默认值。"""
        config = _load_config("dev", "/nonexistent/path.yaml")
        self.assertEqual(config["app_name"], "self_evolving_agent")


class TestOutputFormatting(unittest.TestCase):
    """输出格式化测试。"""

    def test_colorize_non_tty_no_color(self):
        """非TTY环境不加颜色。"""
        with patch("agent_core.cli._COLOR_ENABLED", False):
            result = _colorize("test", "\033[31m")
            self.assertEqual(result, "test")

    def test_colorize_tty_adds_color(self):
        """TTY环境加颜色。"""
        with patch("agent_core.cli._COLOR_ENABLED", True):
            result = _colorize("test", "\033[31m")
            self.assertIn("\033[31m", result)
            self.assertIn("test", result)

    def test_print_success_output(self):
        """成功信息输出。"""
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            _print_success("operation completed")
        output = mock_stdout.getvalue()
        self.assertIn("operation completed", output)

    def test_print_error_output(self):
        """错误信息输出到stderr。"""
        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            _print_error("something failed")
        output = mock_stderr.getvalue()
        self.assertIn("something failed", output)

    def test_print_warning_output(self):
        """警告信息输出。"""
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            _print_warning("caution advised")
        output = mock_stdout.getvalue()
        self.assertIn("caution advised", output)

    def test_print_info_output(self):
        """信息输出。"""
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            _print_info("informational message")
        output = mock_stdout.getvalue()
        self.assertIn("informational message", output)


class TestVersionCommand(unittest.TestCase):
    """version命令测试。"""

    def test_version_outputs_version(self):
        """version命令输出版本号。"""
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            result = main(["version"])
        output = mock_stdout.getvalue()
        self.assertEqual(result, 0)
        self.assertIn("Self-Evolving Agent", output)
        self.assertIn("1.0.0", output)

    def test_version_outputs_python_version(self):
        """version命令输出Python版本。"""
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            main(["version"])
        output = mock_stdout.getvalue()
        # 新格式使用键值对：Python <version>
        self.assertIn("Python", output)
        self.assertIn("3.", output)


class TestStatusCommand(unittest.TestCase):
    """status命令测试。"""

    def test_status_no_agent_running(self):
        """无Agent运行时友好提示并显示日志。"""
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            result = main(["status", "--port", "19999"])
        self.assertEqual(result, 1)
        output = mock_stdout.getvalue()
        # 新行为：友好提示Agent未运行，显示端口号和启动建议
        self.assertIn("not running", output)
        self.assertIn("19999", output)
        self.assertIn("agent start", output)


class TestDashboardCommand(unittest.TestCase):
    """dashboard命令测试。"""

    def test_dashboard_server_lifecycle(self):
        """dashboard server可启动停止。"""
        from agent_core.monitoring.dashboard_server import (
            get_dashboard_server,
        )

        dashboard = get_dashboard_server()
        self.assertFalse(dashboard.is_running())


class TestStartCommand(unittest.TestCase):
    """start命令测试（使用mock避免真实启动）。"""

    def test_start_dispatched_to_handler(self):
        """start命令被正确分发到处理器。"""
        call_args = []

        async def fake_handler(args):
            call_args.append(args)
            return 0

        # 直接替换handlers字典中的引用
        import agent_core.cli as cli_module
        original = cli_module._cmd_start
        cli_module._cmd_start = fake_handler
        try:
            result = main(["start", "--env", "test"])
        finally:
            cli_module._cmd_start = original

        self.assertEqual(result, 0)
        self.assertEqual(len(call_args), 1)
        self.assertEqual(call_args[0].env, "test")

    def test_start_handler_exception_returns_error(self):
        """start处理器异常返回1。"""
        async def failing_handler(args):
            raise RuntimeError("init failed")

        import agent_core.cli as cli_module
        original = cli_module._cmd_start
        cli_module._cmd_start = failing_handler
        try:
            with patch("sys.stderr", new_callable=io.StringIO):
                result = main(["start"])
        finally:
            cli_module._cmd_start = original

        self.assertEqual(result, 1)


class TestMainEntry(unittest.TestCase):
    """主入口测试。"""

    def test_main_no_args_returns_zero(self):
        """无参数返回0。"""
        with patch("sys.stdout", new_callable=io.StringIO):
            result = main([])
        self.assertEqual(result, 0)

    def test_main_unknown_command_returns_error(self):
        """未知命令返回错误。"""
        with patch("sys.stderr", new_callable=io.StringIO):
            # argparse会拒绝未知命令
            with self.assertRaises(SystemExit):
                main(["unknown_command"])

    def test_main_keyboard_interrupt_returns_130(self):
        """Ctrl+C返回130。"""
        with patch(
            "agent_core.cli._cmd_version",
            side_effect=KeyboardInterrupt,
        ):
            with patch("sys.stdout", new_callable=io.StringIO):
                result = main(["version"])
        self.assertEqual(result, 130)


class TestModuleImportable(unittest.TestCase):
    """模块可导入性测试。"""

    def test_agent_core_importable(self):
        """agent_core包可导入。"""
        import agent_core
        self.assertTrue(hasattr(agent_core, "__version__"))

    def test_cli_module_importable(self):
        """cli模块可导入。"""
        from agent_core import cli
        self.assertTrue(hasattr(cli, "main"))

    def test_main_module_importable(self):
        """__main__模块可导入。"""
        from agent_core import __main__
        self.assertTrue(hasattr(__main__, "main"))


if __name__ == "__main__":
    unittest.main()

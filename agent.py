"""
Agent主类

第一性原理：生命体的核心循环是"感知→评估→决策→行动→学习"。
Agent类整合所有模块，实现完整的自驱永续进化循环。

依赖注入规范：
- Agent通过构造函数接收所有依赖，支持Mock注入进行单元测试
- 每个模块依赖接口而非具体实现（IPredictor, IDriveEngine等）
- 默认实现通过工厂方法提供
"""

from __future__ import annotations

import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from core import (
    AgentConfig, LogManager, EventBus, SystemEvent, EventType,
    MemoryEntry, MemoryType, TaskStatus,
    STATE_VECTOR_DIM,
)
from core.interfaces import (
    IPredictor, IDriveEngine, ITaskGenerator,
    IMemory, ISafetyMonitor, IOptimizer, ISensor,
)
from memory import Experience
from memory import ReplayBuffer, ShortTermMemory
from optimizer import OfflineOptimizer
from optimizer.evolution_dynamics import EvolutionDynamics
from optimizer.evolution_roadmap import EvolutionRoadmap

# 易用性阶段4：集成ErrorAdvisor，提供友好错误诊断
try:
    from agent_core.common.error_advisor import ErrorAdvisor
    _error_advisor = ErrorAdvisor()
except ImportError:
    _error_advisor = None


def _format_error(exc: Exception, context: str = "") -> str:
    """格式化错误信息为友好输出（集成ErrorAdvisor）。

    第一性原理：错误信息的价值在于"可操作性"——告诉用户做什么，
    而非仅描述发生了什么。
    """
    if _error_advisor is not None:
        brief = _error_advisor.format_error_brief(exc, color=False)
        return f"{context}: {brief}" if context else brief
    return f"{context}: {exc}" if context else str(exc)


class SelfEvolvingAgent:
    """
    自驱永续进化Agent
    
    核心循环（每步）：
    1. 感知：采集系统指标
    2. 编码：转换为28维状态向量
    3. 预测：多层融合预测下一步状态
    4. 评估：计算预测误差，更新驱力
    5. 决策：生成任务
    6. 行动：执行任务（MVP阶段为模拟）
    7. 学习：在线更新预测模型
    8. 记忆：存储经验
    9. 优化：定期离线复盘
    
    接口契约：
    - predictor: 必须实现 IPredictor 接口
    - drive_engine: 必须实现 IDriveEngine 接口
    - task_generator: 必须实现 ITaskGenerator 接口
    - memory: 必须实现 IMemory 接口
    - safety_monitor: 必须实现 ISafetyMonitor 接口
    """
    
    def __init__(self,
                 config: Optional[AgentConfig] = None,
                 predictor: Optional[IPredictor] = None,
                 drive_engine: Optional[IDriveEngine] = None,
                 task_generator: Optional[ITaskGenerator] = None,
                 memory: Optional[IMemory] = None,
                 safety_monitor: Optional[ISafetyMonitor] = None,
                 optimizer: Optional[IOptimizer] = None,
                 sensors: Optional[list] = None):
        """
        初始化Agent

        Args:
            config: Agent配置，默认创建新实例
            predictor: 预测模型，默认使用 EnsemblePredictor
            drive_engine: 驱力引擎，默认使用 DriveEngine
            task_generator: 任务生成器，默认使用 TaskGenerator
            memory: 记忆系统，默认使用 ShortTermMemory
            safety_monitor: 安全监控，默认使用 ResourceLimiter
            optimizer: 优化器，默认使用 OfflineOptimizer
            sensors: 感知源列表（ISensor接口），默认使用SystemMonitor

        依赖注入示例：
            # 使用默认实现
            agent = SelfEvolvingAgent()

            # 注入自定义预测器
            agent = SelfEvolvingAgent(predictor=MyCustomPredictor())

            # 全部自定义
            agent = SelfEvolvingAgent(
                config=my_config,
                predictor=my_predictor,
                drive_engine=my_drive,
                task_generator=my_tasks,
                memory=my_memory,
                safety_monitor=my_safety,
                optimizer=my_optimizer,
                sensors=[MyCustomSensor()],
            )
        """
        self.config = config or AgentConfig()
        self.running = False
        self.step_count = 0

        # 核心组件
        self.logger = LogManager(
            name=self.config.agent_name,
            level=self.config.log_level,
        )
        self.event_bus = EventBus()

        # 感知模块（通过ISensor接口注入）
        if sensors is not None:
            self.sensors = sensors
        else:
            # 默认感知源（延迟导入避免硬编码依赖）
            from perception import SystemMonitor, AnomalyDetector
            self.sensors = [
                SystemMonitor(),
                AnomalyDetector(window_size=self.config.state_history_window),
            ]

        # 驱力引擎（依赖注入）
        if drive_engine is not None:
            self.drive_engine = drive_engine
        else:
            from drive import DriveEngine
            self.drive_engine = DriveEngine(self.config)

        # 预测模型（依赖注入）
        if predictor is not None:
            self.predictor = predictor
        else:
            from prediction import EnsemblePredictor
            self.predictor = EnsemblePredictor()

        # 记忆系统（依赖注入）
        if memory is not None:
            self.memory = memory
        else:
            from memory import ShortTermMemory
            self.memory = ShortTermMemory(
                max_size=self.config.short_term_memory_size,
                db_path=self.config.long_term_db_path,
            )

        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(
            capacity=self.config.replay_buffer_size,
        )

        # 任务系统（依赖注入）
        if task_generator is not None:
            self.task_generator = task_generator
        else:
            from task import TaskGenerator
            self.task_generator = TaskGenerator()

        # 安全模块（依赖注入）
        if safety_monitor is not None:
            self.safety_monitor = safety_monitor
        else:
            from safety import ResourceLimiter
            self.safety_monitor = ResourceLimiter(self.config)

        # 优化器（依赖注入）
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            from optimizer import OfflineOptimizer
            self.optimizer = OfflineOptimizer(
                min_experiences=50,
                improvement_threshold=0.05,
            )

        # 进化动力学追踪
        self.evolution_dynamics = EvolutionDynamics(max_history=1000)

        # 演进路径规划
        self.evolution_roadmap = EvolutionRoadmap(
            dynamics=self.evolution_dynamics,
            check_interval_seconds=3600.0,
        )

        # === 阶段1：集成打通 — 任务路由器 ===
        # 第一性原理：运动皮层把抽象意图转化为具体肌肉指令。
        # TaskRouter 把 Agent 的抽象任务动作路由到真实执行器。
        # 默认 dry_run=True（安全），生产环境通过 config.execution_dry_run=False 开启真实执行。
        from execution.task_router import TaskRouter
        self.task_router = TaskRouter(
            dry_run=getattr(self.config, "execution_dry_run", True),
            logger=self.logger,
        )

        # === 阶段2：闭环反馈 — 奖励计算器与策略库 ===
        # 第一性原理：多巴胺系统根据"结果与预期的差异"调整未来行为倾向。
        # RewardComputer 把执行结果转化为 reward 信号（基于成功率+指标改善）。
        # StrategyLibrary 记录"故障类型→动作→效果"映射，用UCB平衡探索与利用。
        from learning import RewardComputer, StrategyLibrary
        self.reward_computer = RewardComputer()
        self.strategy_library = StrategyLibrary(exploration_constant=1.0)

        # === 阶段1（新）：目标驱动循环 — 从驱力反射到目标规划 ===
        # 第一性原理：生物体从"刺激-反应"升级到"目标-规划-行动"。
        # GoalDrivenLoop 让Agent能设定长任务目标、分解为子目标、持续推进。
        # 有活跃长任务时走目标驱动；否则回退到原驱力驱动。
        from core.goal_driven_loop import GoalDrivenLoop
        from memory.working_memory import WorkingMemory
        self.working_memory = WorkingMemory(capacity=self.config.working_memory_capacity)
        self.goal_loop = GoalDrivenLoop(
            agent_id=self.config.agent_name,
            working_memory=self.working_memory,
            logger=self.logger,
        )
        # 持久化路径（从config.data_dir派生，不再硬编码）
        # 秩序化阶段1：__init__ 不执行 load()，状态加载移至 load_state()/start()
        self._goal_state_path = Path(self.config.data_dir) / "goals_state.json"

        # === 阶段2（新）：技能库与技能习得 — 从固定动作到可扩展技能 ===
        # 第一性原理：生物体通过"练习→巩固→迁移"习得新技能。
        # SkillLibrary 让Agent能从需求描述生成可执行技能代码、验证、入库、调用。
        # 技能可被目标驱动循环和驱力驱动循环复用，实现技能迁移。
        from core.skill_library import SkillLibrary
        self.skill_library = SkillLibrary(
            skills_dir=f"{self.config.data_dir}/skills",
            max_skills=self.config.skill_max_count,
            logger=self.logger,
            auto_load=False,  # 秩序化阶段1：构造时不加载，由 load_state() 显式加载
        )

        # === 阶段3（新）：多agent协作 — 从单兵到群体协作 ===
        # 第一性原理：社会性生物通过"通信→发现→分派→共识"四要素协作。
        # MultiAgentCollaborator 封装 communication/ 栈，让Agent能发现邻居、
        # 分派任务、处理收到的任务、参与共识决策。
        # 多个agent共享同一MessageBus实例即可协作。
        from core.multi_agent_collaborator import MultiAgentCollaborator
        self.collaborator = MultiAgentCollaborator(
            agent_id=self.config.agent_name,
            heartbeat_interval_seconds=self.config.collaboration_heartbeat_interval_seconds,
            logger=self.logger,
        )
        self._collaboration_step_counter = 0

        # === 阶段4（新）：自驱长任务执行引擎 — 从目标驱动到自驱编排 ===
        # 第一性原理：复杂长任务需要"规划→分派→执行→检查→推进"的闭环。
        # LongTaskOrchestrator 整合目标驱动、技能库、多agent协作，
        # 实现长任务分解、逐步执行、断点续传、失败重试、委派协作。
        from core.long_task_orchestrator import LongTaskOrchestrator
        self.long_task_orchestrator = LongTaskOrchestrator(
            agent_id=self.config.agent_name,
            skill_library=self.skill_library,
            collaborator=self.collaborator,
            checkpoint_dir=f"{self.config.data_dir}/checkpoints",
            logger=self.logger,
        )

        # === 阶段5（新）：自主技能进化闭环 — 从固定技能到自进化技能 ===
        # 第一性原理：技能通过"使用→评估→优化→验证→部署"的闭环不断进化。
        # SkillEvolutionEngine 评估技能性能，自动优化低性能技能，
        # 新版本性能下降时自动回滚，形成自进化闭环。
        from core.skill_evolution import SkillEvolutionEngine
        self.skill_evolution = SkillEvolutionEngine(
            skill_library=self.skill_library,
            logger=self.logger,
        )

        # === 阶段6（新）：多agent自组织与共识 — 从协作到群体智能 ===
        # 第一性原理：群体通过"角色分化→分工协作→共识决策"实现自组织。
        # SelfOrganizationEngine 管理角色（leader/worker/observer），
        # 基于能力组队，通过共识做群体决策。
        from core.self_organization import SelfOrganizationEngine
        self.self_organization = SelfOrganizationEngine(
            collaborator=self.collaborator,
            logger=self.logger,
        )

        # 状态
        self.current_state: Optional[np.ndarray] = None
        self.last_prediction: Optional[np.ndarray] = None
        self.prediction_error_history: list = []
        self._optimization_count = 0
        # 阶段2准备：记录最近一次执行结果，供反馈学习
        self._last_execution_outcomes: list = []

        self.logger.info("Agent initialized", {
            "version": self.config.version,
            "name": self.config.agent_name,
            "execution_dry_run": getattr(self.config, "execution_dry_run", True),
        })
    
    def load_state(self) -> None:
        """显式加载持久化状态（目标状态 + 已有技能）

        第一性原理：构造函数不应有副作用。状态加载是运行时行为，
        应在 start() 中或由调用方显式触发。

        秩序化阶段1：从 __init__ 移出，避免测试实例化Agent时加载污染数据。
        """
        try:
            if self._goal_state_path.exists():
                self.goal_loop.load(self._goal_state_path)
        except Exception as e:
            if self.logger:
                self.logger.warning(_format_error(e, "Failed to load goal state"))
        try:
            self.skill_library._load_from_disk()
        except Exception as e:
            if self.logger:
                self.logger.warning(_format_error(e, "Failed to load skills"))

    def start(self) -> None:
        """启动Agent"""
        self.running = True
        self.event_bus.start()
        for sensor in self.sensors:
            if hasattr(sensor, 'start'):
                sensor.start()
        self.safety_monitor.start()

        # 秩序化阶段1：在启动主循环前加载持久化状态
        if getattr(self.config, 'auto_load_state', True):
            self.load_state()

        self.logger.info("Agent started")
        
        # 注册信号处理（仅在主线程）
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except ValueError:
            pass  # 非主线程，忽略
        
        # 主循环
        try:
            self._main_loop()
        except Exception as e:
            self.logger.error(_format_error(e, "Main loop error"))
        finally:
            self.stop()
    
    def stop(self) -> None:
        """停止Agent"""
        self.running = False
        for sensor in getattr(self, 'sensors', []):
            if hasattr(sensor, 'stop'):
                sensor.stop()
        self.safety_monitor.stop()
        self.event_bus.stop()
        self.logger.info("Agent stopped")
    
    def _signal_handler(self, signum, frame):
        """信号处理"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)
    
    def _main_loop(self) -> None:
        """主循环"""
        while self.running:
            loop_start = time.time()
            
            try:
                self._step()
            except Exception as e:
                self.logger.error(_format_error(e, "Step error"))
            
            self.step_count += 1
            
            # 控制循环频率
            elapsed = time.time() - loop_start
            sleep_time = (self.config.main_loop_interval_ms / 1000.0) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _step(self) -> None:
        """单步执行"""
        # === 1. 感知（从所有sensors采集） ===
        state_vector = None
        anomaly_result = None

        for sensor in self.sensors:
            try:
                perception_input = sensor.read()
                if perception_input is not None and hasattr(sensor, 'encode_to_vector'):
                    perception_output = sensor.encode_to_vector(perception_input)
                    if perception_output is not None and hasattr(perception_output, 'feature_vector'):
                        state_vector = perception_output.feature_vector
                        self.current_state = state_vector
                # 异常检测（如果sensor支持）
                if hasattr(sensor, 'update') and state_vector is not None:
                    anomaly_result = sensor.update(state_vector)
            except Exception:
                self.logger.debug("Snapshot skipped")

        if state_vector is None:
            return

        # === 1.5 数据清洗：过滤NaN/Inf，防止污染后续计算 ===
        # 第一性原理：感觉系统对噪声有阈值过滤，不会把异常信号直接传入皮层。
        if not np.all(np.isfinite(state_vector)):
            self.logger.warning("State vector contains NaN/Inf, replacing with zeros")
            state_vector = np.nan_to_num(state_vector, nan=0.0, posinf=1.0, neginf=0.0)
            self.current_state = state_vector

        # === 2. 预测（多层融合） ===
        if self.last_prediction is not None:
            # 计算预测误差
            prediction_error = float(np.mean((self.last_prediction - state_vector) ** 2))
            # 防止NaN/Inf污染误差历史
            if not np.isfinite(prediction_error):
                prediction_error = 0.0
            self.prediction_error_history.append(prediction_error)
            if len(self.prediction_error_history) > 100:
                self.prediction_error_history.pop(0)
        else:
            prediction_error = 0.0
        
        # 生成新预测（Ensemble: GRU + Transformer + GNN）
        prediction = self.predictor.predict(state_vector)
        self.last_prediction = prediction.value.copy()
        
        # 获取各层状态
        layer_status = self.predictor.get_layer_status() if hasattr(self.predictor, 'get_layer_status') else {}
        
        # === 3. 驱力计算 ===
        drive_state = self.drive_engine.compute(state_vector, prediction_error)

        # === 4. 行动：目标驱动优先，否则驱力驱动 ===
        # 第一性原理（新阶段1）：有长任务目标时走目标驱动，否则回退到驱力驱动。
        tasks = []  # 初始化，供后续日志使用（目标驱动分支不生成tasks）
        if self.goal_loop.has_active_goal():
            # 目标驱动：执行当前子目标的一步
            self._execute_goal_step(state_vector)
        else:
            # 驱力驱动：生成并执行任务（原逻辑）
            tasks = self.task_generator.generate(drive_state, state_vector)
            for task in tasks[:3]:  # 每步最多执行3个任务
                self._execute_task(task)
        
        # === 6. 在线学习 ===
        if self.last_prediction is not None and prediction_error > 0:
            error = self.predictor.update(state_vector, state_vector)
            
            # 存储经验
            exp = Experience(
                state=state_vector,
                action=None,
                reward=-prediction_error,  # 负误差作为奖励
                next_state=state_vector,
            )
            self.replay_buffer.add(exp)
        
        # === 7. 记忆存储 ===
        entry = MemoryEntry(
            content={
                "state": state_vector.tolist(),
                "prediction_error": prediction_error,
                "anomaly_score": anomaly_result.get("anomaly_score", 0) if anomaly_result else 0,
                "layer_weights": {k: v.get('weight', 0) for k, v in layer_status.items() if v} if layer_status else {},
            },
            memory_type=MemoryType.EPISODIC,
            value_score=1.0 - min(prediction_error * 10, 1.0),
            tags=["system_state"],
        )
        self.memory.store(entry)
        
        # === 8. 离线优化（每100步触发一次） ===
        if self.step_count > 0 and self.step_count % 100 == 0:
            if len(self.replay_buffer) >= 50:
                self.logger.info("Triggering offline optimization...")
                opt_result = self.optimizer.optimize(self.predictor, self.replay_buffer)
                self._optimization_count += 1
                self.logger.info(
                    f"Offline optimization: success={opt_result.success}, "
                    f"improvement={opt_result.improvement:.4f}"
                )

                # 记录进化快照
                self._record_evolution_snapshot(prediction_error)

                # 检查演进里程碑
                changed = self.evolution_roadmap.check_milestones()
                for ms in changed:
                    self.logger.info(
                        f"Milestone '{ms.name}' status changed to {ms.status.value}"
                    )

                # 获取策略推荐
                strategy = self.evolution_roadmap.recommend_strategy()
                self.logger.info(
                    f"Strategy recommendation: {strategy['primary_strategy']} - "
                    f"{strategy['primary_reason']}"
                )

                # 检查是否在轨道上
                on_track, track_reason = self.evolution_roadmap.is_on_track()
                if not on_track:
                    self.logger.warning(f"Evolution off track: {track_reason}")
        
        # === 9. 日志和事件 ===
        if self.step_count % 10 == 0:
            layer_info = ""
            if layer_status:
                layer_info = (f" Layers=[GRU:{layer_status.get('gru', {}).get('weight', 0):.2f} "
                             f"TR:{layer_status.get('transformer', {}).get('weight', 0):.2f} "
                             f"GNN:{layer_status.get('gnn', {}).get('weight', 0):.2f}]")

            self.logger.info(
                f"Step {self.step_count}: "
                f"CPU={state_vector[0]:.2f} "
                f"MEM={state_vector[10]:.2f} "
                f"Err={prediction_error:.4f} "
                f"Tasks={len(tasks)}"
                f"{layer_info}",
                {
                    "step": self.step_count,
                    "prediction_error": f"{prediction_error:.4f}",
                    "anomaly": str(anomaly_result.get("is_anomaly", False) if anomaly_result else False),
                    "emergency": str(drive_state.emergency),
                }
            )

        # === 10. 多agent协作（阶段3新） ===
        # 第一性原理：社会性生物周期性发出存在信号，并响应群体消息。
        # 每步处理收到的协作消息；每3步广播一次心跳（受heartbeat_interval控制）。
        try:
            self.collaborator.process_messages(max_messages=50)
            # 计算当前负载（CPU使用率作为近似）
            load_score = float(state_vector[0]) / 100.0 if len(state_vector) > 0 else 0.0
            load_score = max(0.0, min(1.0, load_score))
            self.collaborator.broadcast_heartbeat(load_score=load_score)
            self._collaboration_step_counter += 1
        except Exception as e:
            self.logger.debug(_format_error(e, "Collaboration step skipped"))
        
        # 发布系统状态事件
        self.event_bus.publish(SystemEvent(
            event_type=EventType.SYSTEM_STATE,
            payload={
                "step": self.step_count,
                "state": state_vector.tolist(),
                "prediction_error": prediction_error,
                "layer_weights": {k: v.get('weight', 0) for k, v in layer_status.items()} if layer_status else {},
                "drive_state": {
                    k.value: v.intensity 
                    for k, v in drive_state.drives.items()
                },
            },
        ))
    
    def _execute_task(self, task) -> None:
        """
        执行任务（阶段2：闭环反馈 — 执行→采集结果→计算reward→记录策略→存经验）

        第一性原理：生物体的学习基于"行动→结果→强化/抑制"的闭环。
        多巴胺系统根据结果与预期的差异调整未来行为倾向。

        闭环链路：
            1. 采集执行前指标快照 (before)
            2. TaskRouter 路由到执行器执行
            3. 采集执行后指标快照 (after)
            4. RewardComputer 基于 (success, before→after delta) 计算 reward
            5. StrategyLibrary 记录 (故障类型, action, success, reward)
            6. 存 Experience 到 replay_buffer（带真实 reward）
            7. 更新 task 对象
        """
        from learning import MetricsSnapshot

        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        self.logger.info(f"Executing task: {task.name} - {task.description}")

        # 1. 采集执行前指标快照
        before_snapshot = MetricsSnapshot.from_state_vector(self.current_state)

        # 2. 路由到真实执行器
        outcome = self.task_router.route(task, self.current_state)

        # 3. 采集执行后指标快照（同步执行，after≈before；真实执行后指标可能变化）
        after_snapshot = MetricsSnapshot.from_state_vector(self.current_state)

        # 4. 计算真实 reward（基于成功率 + 指标改善）
        reward = self.reward_computer.compute(
            success=outcome.success,
            executed=outcome.executed,
            before=before_snapshot,
            after=after_snapshot,
            task_action=getattr(task, "action", task.name),
        )

        # 5. 记录到策略库（供未来决策参考）
        fault_type = self.strategy_library.classify_fault(before_snapshot)
        self.strategy_library.record(
            fault_type=fault_type,
            action=getattr(task, "action", task.name),
            success=outcome.success,
            reward=reward,
        )

        # 6. 存 Experience 到 replay_buffer（带真实 reward，供离线优化器学习）
        if self.current_state is not None:
            exp = Experience(
                state=self.current_state.copy(),
                action=getattr(task, "action", task.name),
                reward=reward,  # 真实 reward（不再是固定0.1）
                next_state=self.current_state.copy(),  # 同步执行，next≈current
            )
            self.replay_buffer.add(exp)

        # 7. 更新 task 对象
        if outcome.success:
            task.status = TaskStatus.COMPLETED
        else:
            task.status = TaskStatus.FAILED
        task.completed_at = time.time()
        task.reward = reward  # 真实 reward
        task.actual_outcome = (
            f"executor={outcome.executor} success={outcome.success} reward={reward:.3f} "
            f"fault={fault_type} cmd='{outcome.command[:60]}' rc={outcome.return_code}"
        )
        task.metadata["execution_outcome"] = {
            "executor": outcome.executor,
            "executed": outcome.executed,
            "success": outcome.success,
            "command": outcome.command,
            "return_code": outcome.return_code,
            "duration_ms": outcome.duration_ms,
            "reward": reward,
            "fault_type": fault_type,
            "before_cpu": before_snapshot.cpu_percent,
            "after_cpu": after_snapshot.cpu_percent,
        }

        # 记录执行结果（供阶段2反馈学习）
        self._last_execution_outcomes.append(outcome)
        if len(self._last_execution_outcomes) > 100:
            self._last_execution_outcomes.pop(0)

        self.task_generator.complete_task(task.id, task.reward)

    # ========================================================================
    # 阶段1（新）：目标驱动 — 长任务设定与执行
    # ========================================================================

    def set_long_task(self,
                      description: str,
                      sub_tasks: List[str],
                      priority: float = 0.8,
                      deadline_hours: float = 1.0,
                      steps_per_subgoal: int = 10) -> str:
        """
        设定长任务目标并分解为子目标序列

        第一性原理：长任务无法一步完成，必须分解为可执行的子目标序列。
        Agent会按顺序推进每个子目标，支持断点续传。

        Args:
            description: 长任务描述
            sub_tasks: 子任务描述列表（按执行顺序）
            priority: 优先级 0-1
            deadline_hours: 截止时间（小时）
            steps_per_subgoal: 每个子目标预期执行步数

        Returns:
            父目标ID
        """
        goal_id = self.goal_loop.set_long_task(
            description=description,
            sub_tasks=sub_tasks,
            priority=priority,
            deadline_hours=deadline_hours,
            steps_per_subgoal=steps_per_subgoal,
        )
        # 持久化（断点续传）
        self.goal_loop.persist(self._goal_state_path)
        return goal_id

    def _execute_goal_step(self, state_vector: np.ndarray) -> None:
        """
        执行一步目标驱动

        第一性原理：把子目标描述映射为具体动作，通过TaskRouter执行，
        并根据执行结果更新子目标进度。

        子目标描述→动作的映射策略：
        - 包含"cpu" → reduce_cpu_load
        - 包含"memory"/"内存" → free_memory
        - 包含"disk"/"磁盘" → free_disk_space
        - 包含"optimize"/"优化" → optimize_resources
        - 包含"sample"/"采样" → increase_sampling_rate
        - 其他 → 通用执行（log_only）
        """
        def action_executor(description: str, sv: np.ndarray) -> Dict[str, Any]:
            """子目标→动作执行回调"""
            # 子目标描述→动作映射
            desc_lower = description.lower()
            if "cpu" in desc_lower:
                action = "reduce_cpu_load"
            elif "memory" in desc_lower or "内存" in description:
                action = "free_memory"
            elif "disk" in desc_lower or "磁盘" in description:
                action = "free_disk_space"
            elif "optimize" in desc_lower or "优化" in description:
                action = "optimize_resources"
            elif "sample" in desc_lower or "采样" in description:
                action = "increase_sampling_rate"
            else:
                action = "log_only"

            # 构造任务对象并路由执行
            from core import Task, TaskPriority, TaskStatus, DriveType
            task = Task(
                id=f"goal_task_{int(time.time()*1000)}",
                name=f"goal_step_{action}",
                description=description,
                priority=TaskPriority.NORMAL,
                drive_source=DriveType.SAFETY,
                action=action,
            )
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()

            # 复用阶段1-2的执行+反馈闭环
            from learning import MetricsSnapshot
            before = MetricsSnapshot.from_state_vector(sv)
            outcome = self.task_router.route(task, sv)
            after = MetricsSnapshot.from_state_vector(sv)
            reward = self.reward_computer.compute(
                success=outcome.success, executed=outcome.executed,
                before=before, after=after, task_action=action,
            )
            fault_type = self.strategy_library.classify_fault(before)
            self.strategy_library.record(
                fault_type=fault_type, action=action,
                success=outcome.success, reward=reward,
            )
            # 存经验
            exp = Experience(
                state=sv.copy(), action=action, reward=reward,
                next_state=sv.copy(),
            )
            self.replay_buffer.add(exp)

            task.reward = reward
            task.status = TaskStatus.COMPLETED if outcome.success else TaskStatus.FAILED
            self._last_execution_outcomes.append(outcome)
            if len(self._last_execution_outcomes) > 100:
                self._last_execution_outcomes.pop(0)

            return {"success": outcome.success, "reward": reward, "outcome": outcome}

        # 执行目标驱动一步
        result = self.goal_loop.execute_step(state_vector, action_executor)

        # 持久化目标状态（断点续传）
        self.goal_loop.persist(self._goal_state_path)

        if self.logger:
            subgoal = self.goal_loop.get_current_subgoal()
            if result.get("status") == "executing":
                self.logger.info(
                    f"Goal step: {result.get('subgoal_description','')} "
                    f"progress={result.get('progress',0):.1%} "
                    f"step={result.get('step',0)}/{result.get('expected_steps',0)}"
                )

    def get_goal_status(self) -> Dict[str, Any]:
        """获取目标驱动状态"""
        return self.goal_loop.get_status()

    # ========================================================================
    # 阶段2（新）：技能习得 — 从需求生成技能、调用技能
    # ========================================================================

    def learn_skill(self,
                    requirement: str,
                    skill_name: Optional[str] = None) -> Dict[str, Any]:
        """
        习得新技能

        第一性原理：技能"迁移"——从自然语言需求生成可执行技能。
        Agent 通过"练习"（生成代码+验证）"巩固"（入库）新技能。

        Args:
            requirement: 需求描述（自然语言）
            skill_name: 指定技能名，None则自动从需求提取

        Returns:
            {"success": bool, "skill_name": str, "reason": str, "validated": bool}
        """
        result = self.skill_library.learn_from_requirement(
            requirement=requirement,
            skill_name=skill_name,
        )
        if self.logger:
            self.logger.info(
                f"Learn skill: requirement='{requirement[:50]}' "
                f"success={result['success']} name={result.get('skill_name','')}"
            )
        return result

    def use_skill(self, name: str, **kwargs) -> Dict[str, Any]:
        """
        调用已习得的技能

        第一性原理：技能"巩固"——每次成功调用强化技能，失败降低成功率。
        长期低成功率的技能会被自动淘汰。

        Args:
            name: 技能名
            **kwargs: 传给技能 execute 函数的参数

        Returns:
            {"success": bool, "result": Any, "duration_ms": float, "error": str}
        """
        result = self.skill_library.invoke(name, **kwargs)
        if self.logger:
            self.logger.info(
                f"Use skill: {name} success={result['success']} "
                f"duration={result['duration_ms']:.2f}ms"
            )
        return result

    def get_skill_status(self) -> Dict[str, Any]:
        """获取技能库状态"""
        return self.skill_library.get_stats()

    # ========================================================================
    # 阶段3（新）：多agent协作 — 任务分派与共识
    # ========================================================================

    def register_capability(self, task_type: str,
                            handler: Callable) -> None:
        """
        注册本agent能处理的任务类型

        第一性原理：个体声明自己的专长，群体据此分派任务。
        其他agent发现本agent具备该能力后，可把任务分派过来。

        Args:
            task_type: 任务类型标识
            handler: 处理函数 handler(payload: dict) -> result
        """
        self.collaborator.register_capability(task_type, handler)

    def collaborate(self,
                    task_type: str,
                    payload: Dict[str, Any],
                    required_capability: Optional[str] = None,
                    on_result: Optional[Callable] = None) -> Tuple[str, Optional[AgentMessage]]:
        """
        分派任务给最佳邻居agent

        第一性原理：当本agent无法独立完成或负载过高时，把任务分派给
        能力匹配且负载最低的邻居。支持结果回调。

        Args:
            task_type: 任务类型
            payload: 任务数据
            required_capability: 所需能力（None则不限制）
            on_result: 结果回调 callback(result_dict)

        Returns:
            (task_id, dispatch_message) — 若无可用邻居，dispatch_message为None
        """
        return self.collaborator.dispatch_task(
            task_type=task_type,
            payload=payload,
            required_capability=required_capability,
            on_result=on_result,
        )

    def propose_consensus(self,
                          proposal_id: str,
                          value: Dict[str, Any],
                          participant_ids: List[str],
                          on_commit: Optional[Callable] = None):
        """
        发起共识提案

        第一性原理：群体决策需要多数同意。2PC让多agent对决策达成一致。

        Args:
            proposal_id: 提案ID（唯一）
            value: 提案内容
            participant_ids: 参与者ID列表
            on_commit: 通过后的回调

        Returns:
            prepare 消息
        """
        return self.collaborator.propose_consensus(
            proposal_id=proposal_id,
            value=value,
            participant_ids=participant_ids,
            on_commit=on_commit,
        )

    def get_collaboration_status(self) -> Dict[str, Any]:
        """获取多agent协作状态"""
        return self.collaborator.get_status()

    # ========================================================================
    # 阶段4（新）：自驱长任务执行引擎
    # ========================================================================

    def start_long_task(self,
                        description: str,
                        phases: List[List[Dict[str, Any]]],
                        metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        启动自驱长任务

        第一性原理：长任务分解为阶段和步骤，每步可本地执行（用技能）
        或委派（给其他agent），支持断点续传和失败重试。

        Args:
            description: 长任务描述
            phases: 阶段列表，每个阶段是步骤dict列表
            metadata: 元数据

        Returns:
            task_id
        """
        return self.long_task_orchestrator.start_task(
            description=description,
            phases=phases,
            metadata=metadata,
        )

    def execute_long_task_step(self) -> Dict[str, Any]:
        """
        执行长任务的下一步

        第一性原理：自驱执行——每步自动决定执行方式，执行后检查结果，
        失败则重试，重试耗尽则降级跳过，自动推进。

        Returns:
            执行结果 {"status": str, "step": dict, "result": Any, "error": str}
        """
        return self.long_task_orchestrator.execute_next()

    def run_long_task(self, max_steps: int = 1000) -> Dict[str, Any]:
        """
        运行长任务直到完成或达到步数上限

        第一性原理：自驱永续——持续执行直到任务完成，无需外部驱动。

        Args:
            max_steps: 最大执行步数（防止无限循环）

        Returns:
            最终进度
        """
        for _ in range(max_steps):
            result = self.long_task_orchestrator.execute_next()
            if result["status"] in ("task_completed", "no_task", "not_running"):
                break
        return self.long_task_orchestrator.get_progress()

    def get_long_task_progress(self) -> Dict[str, Any]:
        """获取长任务进度"""
        return self.long_task_orchestrator.get_progress()

    # ========================================================================
    # 阶段5（新）：自主技能进化闭环
    # ========================================================================

    def evolve_skills(self) -> Dict[str, Any]:
        """
        运行一个技能进化周期

        第一性原理：技能通过"使用→评估→优化→验证→部署"的闭环不断进化。
        自动评估所有技能性能，优化低性能技能，回滚退化版本。

        Returns:
            {"evaluated": int, "optimized": int, "rolled_back": int, "details": list}
        """
        return self.skill_evolution.run_evolution_cycle()

    def evaluate_skill(self, skill_name: str) -> Optional[Dict[str, Any]]:
        """评估单个技能性能"""
        eval_result = self.skill_evolution.evaluate(skill_name)
        if eval_result is None:
            return None
        return {
            "skill_name": eval_result.skill_name,
            "version": eval_result.version,
            "success_rate": eval_result.success_rate,
            "invoke_count": eval_result.invoke_count,
            "avg_duration_ms": eval_result.avg_duration_ms,
            "score": eval_result.score,
            "verdict": eval_result.verdict,
            "issues": eval_result.issues,
        }

    def get_skill_evolution_history(self) -> List[Dict[str, Any]]:
        """获取技能进化历史"""
        return self.skill_evolution.get_evolution_history()

    # ========================================================================
    # 阶段6（新）：多agent自组织与共识 — 角色分化+组队+共识决策
    # ========================================================================

    def assign_role(self, role: str, reason: str = "") -> Dict[str, Any]:
        """
        分配/变更自身角色

        第一性原理：群体中个体根据能力、负载和群体需要承担不同角色。
        角色分化是分工协作的前提。

        Args:
            role: 角色名（follower/candidate/leader/worker/observer）
            reason: 变更原因

        Returns:
            {"role": str, "reason": str, "changed": bool}
        """
        from core.self_organization import AgentRole
        try:
            role_enum = AgentRole(role)
        except ValueError:
            return {"role": role, "reason": reason, "changed": False,
                    "error": f"未知角色: {role}"}
        old_role = self.self_organization.get_role().value
        self.self_organization.assign_role(role_enum, reason=reason)
        return {
            "role": role_enum.value,
            "previous_role": old_role,
            "reason": reason,
            "changed": old_role != role_enum.value,
        }

    def form_team(self,
                  task_description: str,
                  required_capabilities: List[str],
                  team_members: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        基于能力组队

        第一性原理：群体根据任务需求自动组队，选择具备所需能力的成员。
        组队后本agent自动成为leader。

        Args:
            task_description: 任务描述
            required_capabilities: 所需能力列表
            team_members: 指定成员列表，None则自动从邻居中选择

        Returns:
            {"team_id": str, "leader_id": str, "members": list, "capabilities": list}
        """
        team = self.self_organization.form_team(
            task_description=task_description,
            required_capabilities=required_capabilities,
            team_members=team_members,
        )
        return {
            "team_id": team.team_id,
            "leader_id": team.leader_id,
            "members": team.members,
            "capabilities": team.capabilities,
            "task_description": team.task_description,
        }

    def propose_decision(self,
                         decision_id: str,
                         proposal: Dict[str, Any],
                         participant_ids: Optional[List[str]] = None,
                         on_approved: Optional[Callable] = None) -> Dict[str, Any]:
        """
        发起共识决策（仅leader可发起）

        第一性原理：重要决策需要群体共识。Leader发起提案，Workers投票，
        达到quorum则通过，否则否决。

        Args:
            decision_id: 决策ID（唯一）
            proposal: 提案内容
            participant_ids: 参与者列表，None则用团队成员
            on_approved: 通过后的回调 callback(proposal_value)

        Returns:
            {"decision_id": str, "status": str, "proposal": dict}
        """
        return self.self_organization.propose_decision(
            decision_id=decision_id,
            proposal=proposal,
            participant_ids=participant_ids,
            on_approved=on_approved,
        )

    def vote(self, decision_id: str, vote_value: bool) -> Dict[str, Any]:
        """
        对决策投票

        第一性原理：个体根据自身判断对提案投票，群体通过多数达成共识。

        Args:
            decision_id: 决策ID
            vote_value: True=赞成, False=反对

        Returns:
            {"decision_id": str, "vote": bool, "status": str}
        """
        return self.self_organization.vote(decision_id, vote_value)

    def run_organization_cycle(self) -> Dict[str, Any]:
        """
        运行一个自组织周期

        第一性原理：群体周期性自组织——检查leader存活、触发选举、
        处理决策投票、更新角色。Leader广播心跳，Follower检查leader存活。

        Returns:
            {"role": str, "leader": str, "team": str, "decisions": int}
        """
        return self.self_organization.run_organization_cycle()

    def get_organization_status(self) -> Dict[str, Any]:
        """获取自组织状态"""
        return self.self_organization.get_status()

    def step(self) -> Dict[str, Any]:
        """执行单步并返回状态摘要

        Precondition:
            - Agent 已初始化
        Postcondition:
            - step_count 增加1
            - 返回包含当前状态的字典
        """
        try:
            self._step()
            self.step_count += 1
            return self.get_status()
        except Exception as e:
            return {"error": _format_error(e, "step failed")}

    def _record_evolution_snapshot(self, current_mse: float) -> None:
        """记录进化快照"""
        try:
            n_params = 0
            n_layers = 1
            hidden_dim = 16
            if hasattr(self.predictor, 'get_params'):
                params = self.predictor.get_params()
                n_params = sum(p.size for p in params.values())
            if hasattr(self.predictor, 'num_layers'):
                n_layers = self.predictor.num_layers
            if hasattr(self.predictor, 'hidden_dim'):
                hidden_dim = self.predictor.hidden_dim

            # 测量预测延迟
            import time
            t0 = time.time()
            if self.current_state is not None:
                self.predictor.predict(self.current_state)
            latency_ms = (time.time() - t0) * 1000.0

            self.evolution_dynamics.record(
                prediction_mse=current_mse,
                n_parameters=n_params,
                n_layers=n_layers,
                hidden_dim=hidden_dim,
                prediction_latency_ms=latency_ms,
                optimization_count=self._optimization_count,
                total_experiences=len(self.replay_buffer),
            )
        except Exception as e:
            self.logger.error(_format_error(e, "Failed to record evolution snapshot"))

    def get_status(self) -> Dict[str, Any]:
        """获取Agent状态"""
        return {
            "running": self.running,
            "step_count": self.step_count,
            "replay_buffer_size": len(self.replay_buffer),
            "memory_size": len(self.memory) if hasattr(self.memory, '__len__') else 0,
            "pending_tasks": len(self.task_generator.get_pending_tasks()) if hasattr(self.task_generator, 'get_pending_tasks') else 0,
            "optimization_count": self._optimization_count,
            "evolution_snapshots": len(self.evolution_dynamics.snapshots),
            "execution_stats": self.task_router.get_stats() if hasattr(self, 'task_router') else {},
            "last_execution_count": len(self._last_execution_outcomes) if hasattr(self, '_last_execution_outcomes') else 0,
            "strategy_stats": self.strategy_library.get_stats() if hasattr(self, 'strategy_library') else {},
            "goal_status": self.goal_loop.get_status() if hasattr(self, 'goal_loop') else {},
            "working_memory_stats": self.working_memory.get_stats() if hasattr(self, 'working_memory') else {},
            "skill_stats": self.skill_library.get_stats() if hasattr(self, 'skill_library') else {},
            "collaboration_status": self.collaborator.get_status() if hasattr(self, 'collaborator') else {},
            "long_task_status": self.long_task_orchestrator.get_status() if hasattr(self, 'long_task_orchestrator') else {},
            "skill_evolution_status": self.skill_evolution.get_status() if hasattr(self, 'skill_evolution') else {},
            "organization_status": self.self_organization.get_status() if hasattr(self, 'self_organization') else {},
        }


# 秩序化阶段5：入口秩序化
# 唯一入口为 main.py，agent.py 不再定义 main()，避免重复入口混淆。
# 启动方式：python main.py

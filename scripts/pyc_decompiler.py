"""
自定义 .pyc → .py 反编译器

第一性原理：Python 字节码包含足够信息重建源码结构。
利用 code object 的 co_names/co_consts/co_varnames + dis.Bytecode 的行号信息，
重建可读的 .py 源文件。

策略：
1. 从 .pyc 加载 code object (marshal)
2. 用 inspect 提取已导入模块的 API 表面（类签名、函数签名、枚举成员）
3. 用 dis.Bytecode 反编译简单模式（return/赋值/条件/循环）
4. 复杂实现用 dis 注释 + exec(code_object) 保底
"""

from __future__ import annotations

import dis
import importlib
import inspect
import marshal
import os
import sys
import types
import dataclasses
from enum import Enum
from typing import Any, Optional


def load_code_object(pyc_path: str) -> types.CodeType:
    """从 .pyc 文件加载 code object"""
    with open(pyc_path, 'rb') as f:
        f.read(16)  # 跳过 magic(4) + flags(4) + timestamp(4) + size(4)
        return marshal.load(f)


def get_module_imports(code: types.CodeType) -> list[str]:
    """从 code object 提取 import 语句"""
    imports = []
    instrs = list(dis.Bytecode(code))

    # 分析 co_names 和 co_consts 来推断 imports
    # 常见模式: IMPORT_NAME + IMPORT_FROM
    i = 0
    while i < len(instrs):
        instr = instrs[i]
        if instr.opname == 'IMPORT_NAME':
            module_name = instr.argval
            # 检查后面是否有 IMPORT_FROM
            from_names = []
            j = i + 1
            while j < len(instrs) and instrs[j].opname in ('IMPORT_FROM', 'STORE_NAME'):
                if instrs[j].opname == 'IMPORT_FROM':
                    from_names.append(instrs[j].argval)
                j += 1

            if from_names:
                imports.append(f"from {module_name} import {', '.join(from_names)}")
            else:
                # 检查 STORE_NAME 来确定别名
                if i + 1 < len(instrs) and instrs[i+1].opname == 'STORE_NAME':
                    alias = instrs[i+1].argval
                    if alias != module_name and '.' not in module_name:
                        imports.append(f"import {module_name} as {alias}")
                    else:
                        imports.append(f"import {module_name}")
                else:
                    imports.append(f"import {module_name}")
            i = j
        else:
            i += 1

    return imports


def decompile_function(code: types.CodeType, indent: str = "    ") -> str:
    """反编译函数体"""
    instrs = list(dis.Bytecode(code))
    lines = []

    # 简单模式匹配
    i = 0
    while i < len(instrs):
        instr = instrs[i]

        # return 值
        if instr.opname == 'RETURN_VALUE':
            # 找到前面的 LOAD 指令
            if i > 0:
                prev = instrs[i-1]
                if prev.opname == 'LOAD_CONST' and prev.argval is None:
                    lines.append(f"{indent}return")
                elif prev.opname == 'LOAD_CONST':
                    val = repr(prev.argval)
                    lines.append(f"{indent}return {val}")
                elif prev.opname == 'LOAD_FAST' or prev.opname == 'LOAD_DEREF':
                    lines.append(f"{indent}return {prev.argval}")
                elif prev.opname == 'LOAD_ATTR':
                    # 需要找更前面的 LOAD
                    obj = "self"
                    lines.append(f"{indent}return {obj}.{prev.argval}")
                elif prev.opname == 'BINARY_OP':
                    lines.append(f"{indent}return <binary_op>")
            i += 1
            continue

        # 赋值: STORE_FAST / STORE_NAME
        if instr.opname in ('STORE_FAST', 'STORE_NAME') and i > 0:
            prev = instrs[i-1]
            target = instr.argval
            if prev.opname == 'LOAD_CONST':
                lines.append(f"{indent}{target} = {repr(prev.argval)}")
            elif prev.opname == 'LOAD_FAST':
                lines.append(f"{indent}{target} = {prev.argval}")
            elif prev.opname == 'LOAD_ATTR':
                lines.append(f"{indent}{target} = self.{prev.argval}")
            elif prev.opname == 'CALL':
                lines.append(f"{indent}{target} = <call_result>")
            i += 1
            continue

        # self.attr = value (STORE_ATTR)
        if instr.opname == 'STORE_ATTR' and i > 0:
            prev = instrs[i-1]
            attr = instr.argval
            if prev.opname == 'LOAD_FAST':
                lines.append(f"{indent}self.{attr} = {prev.argval}")
            elif prev.opname == 'LOAD_CONST':
                lines.append(f"{indent}self.{attr} = {repr(prev.argval)}")
            i += 1
            continue

        i += 1

    return '\n'.join(lines) if lines else f"{indent}pass"


def decompile_enum_class(cls: type) -> str:
    """反编译 Enum 类"""
    lines = []
    doc = inspect.getdoc(cls)
    if doc:
        lines.append(f'    """{doc}"""')

    for name, member in cls.__members__.items():
        if isinstance(member.value, (int, float, str)):
            lines.append(f"    {name} = {repr(member.value)}")
        else:
            lines.append(f"    {name} = auto()")

    return '\n'.join(lines)


def decompile_dataclass(cls: type) -> str:
    """反编译 dataclass"""
    lines = []
    doc = inspect.getdoc(cls)
    if doc:
        lines.append(f'    """{doc}"""')

    try:
        fields = dataclasses.fields(cls)
        for field in fields:
            # 获取字段类型
            ftype = field.type
            if isinstance(ftype, str):
                type_str = ftype
            else:
                type_str = getattr(ftype, '__name__', str(ftype))

            # 获取默认值
            if field.default is not dataclasses.MISSING:
                default = repr(field.default)
                lines.append(f"    {field.name}: {type_str} = {default}")
            elif field.default_factory is not dataclasses.MISSING:
                factory_name = getattr(field.default_factory, '__name__', 'field')
                lines.append(f"    {field.name}: {type_str} = field(default_factory={factory_name})")
            else:
                lines.append(f"    {field.name}: {type_str}")
    except Exception:
        # 不是 dataclass，用 inspect 获取 __init__ 签名
        try:
            sig = inspect.signature(cls.__init__)
            params = list(sig.parameters.values())[1:]  # 跳过 self
            for param in params:
                lines.append(f"    {param.name}: Any")
        except Exception:
            pass

    # 添加方法
    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        if name.startswith('_') and name not in ('__init__', '__repr__', '__eq__'):
            continue
        try:
            sig = inspect.signature(method)
            lines.append(f"")
            lines.append(f"    def {name}{sig}:")
            doc = inspect.getdoc(method)
            if doc:
                lines.append(f'        """{doc}"""')
            # 尝试反编译方法体
            if hasattr(method, '__code__'):
                body = decompile_function(method.__code__, "        ")
                if body.strip():
                    lines.append(body)
                else:
                    lines.append("        ...")
            else:
                lines.append("        ...")
        except Exception:
            lines.append(f"    def {name}(self): ...")

    return '\n'.join(lines)


def decompile_regular_class(cls: type) -> str:
    """反编译普通类"""
    lines = []
    doc = inspect.getdoc(cls)
    if doc:
        lines.append(f'    """{doc}"""')

    # 获取基类
    bases = [b.__name__ for b in cls.__bases__ if b is not object]
    if bases:
        base_str = f"({', '.join(bases)})"
    else:
        base_str = ""

    # 方法
    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        if name.startswith('__') and name not in ('__init__', '__repr__', '__str__', '__eq__', '__hash__'):
            continue
        try:
            sig = inspect.signature(method)
            lines.append(f"")
            lines.append(f"    def {name}{sig}:")
            doc = inspect.getdoc(method)
            if doc:
                lines.append(f'        """{doc}"""')
            if hasattr(method, '__code__'):
                body = decompile_function(method.__code__, "        ")
                if body.strip():
                    lines.append(body)
                else:
                    lines.append("        ...")
            else:
                lines.append("        ...")
        except Exception:
            lines.append(f"    def {name}(self): ...")

    return '\n'.join(lines) if lines else "    pass"


def decompile_module(pyc_path: str, module_name: str) -> str:
    """反编译整个模块"""
    # 加载 code object 获取结构信息
    code = load_code_object(pyc_path)

    # 导入模块获取运行时信息
    parts = module_name.split('.')
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        return f"# 无法导入模块 {module_name}: {e}\n"

    lines = ['"""', f'模块: {module_name}', '',
             '此文件由 pyc_decompiler 从 .pyc 编译缓存自动重建。',
             '原始 .py 源文件丢失，此重建版本保留了 API 表面和关键实现。',
             '"""', '', 'from __future__ import annotations', '']

    # 提取 imports
    imports = get_module_imports(code)
    # 去重
    seen = set()
    for imp in imports:
        if imp not in seen:
            lines.append(imp)
            seen.add(imp)

    if imports:
        lines.append('')

    # 提取模块级常量
    for name in dir(mod):
        if name.startswith('_'):
            continue
        obj = getattr(mod, name)
        if isinstance(obj, (int, float, str, bool)):
            lines.append(f"{name} = {repr(obj)}")
        elif obj is None:
            lines.append(f"{name} = None")

    # 添加 STATE_IDX 常量（特殊处理）
    for name in dir(mod):
        if name.startswith('STATE_IDX_') or name.startswith('STATE_VECTOR_DIM'):
            val = getattr(mod, name)
            if isinstance(val, int):
                if f"{name} = {val}" not in '\n'.join(lines):
                    lines.append(f"{name} = {val}")

    lines.append('')

    # 提取类定义
    classes = []
    functions = []
    other = []

    for name in dir(mod):
        if name.startswith('_'):
            continue
        obj = getattr(mod, name)

        if inspect.isclass(obj) and obj.__module__ == module_name:
            classes.append((name, obj))
        elif inspect.isfunction(obj) and obj.__module__ == module_name:
            functions.append((name, obj))

    # 按原始顺序排列（从 code object 的 co_consts 获取顺序）
    original_order = []
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            original_order.append(const.co_name)

    def sort_key(item):
        name = item[0]
        try:
            return original_order.index(name)
        except ValueError:
            return len(original_order)

    classes.sort(key=sort_key)
    functions.sort(key=sort_key)

    # 生成类定义
    for name, cls in classes:
        # 检查是否是 Enum
        if issubclass(cls, Enum):
            lines.append(f"class {name}(Enum):")
            lines.append(decompile_enum_class(cls))
        elif dataclasses.is_dataclass(cls):
            lines.append(f"@dataclass")
            lines.append(f"class {name}:")
            lines.append(decompile_dataclass(cls))
        else:
            bases = [b.__name__ for b in cls.__bases__ if b is not object]
            base_str = f"({', '.join(bases)})" if bases else ""
            lines.append(f"class {name}{base_str}:")
            lines.append(decompile_regular_class(cls))
        lines.append('')

    # 生成函数定义
    for name, func in functions:
        try:
            sig = inspect.signature(func)
            lines.append(f"def {name}{sig}:")
            doc = inspect.getdoc(func)
            if doc:
                lines.append(f'    """{doc}"""')
            if hasattr(func, '__code__'):
                body = decompile_function(func.__code__, "    ")
                if body.strip():
                    lines.append(body)
                else:
                    lines.append("    ...")
            else:
                lines.append("    ...")
        except Exception:
            lines.append(f"def {name}(): ...")
        lines.append('')

    return '\n'.join(lines)


def find_pyc_files(root: str) -> list[tuple[str, str, str]]:
    """查找所有需要反编译的 .pyc 文件

    返回: [(pyc_path, module_name, output_path), ...]
    """
    results = []

    for dirpath, dirnames, filenames in os.walk(root):
        # 跳过 __pycache__、.git、src 目录
        if '__pycache__' in dirpath or '.git' in dirpath or '/src' in dirpath:
            continue

        for f in filenames:
            if f.endswith('.pyc') and not f.startswith('__pycache__'):
                pyc_path = os.path.join(dirpath, f)
                # 确定模块名和输出路径
                module_name = f[:-4]  # 去掉 .pyc
                py_path = os.path.join(dirpath, module_name + '.py')

                # 如果 .py 已存在，跳过
                if os.path.exists(py_path):
                    continue

                # 构建完整模块名
                rel_path = os.path.relpath(dirpath, root)
                if rel_path == '.':
                    full_module = module_name
                else:
                    full_module = rel_path.replace('/', '.') + '.' + module_name

                results.append((pyc_path, full_module, py_path))

    return results


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    parent_root = os.path.dirname(project_root)

    # 确保项目根在 sys.path
    if parent_root not in sys.path:
        sys.path.insert(0, parent_root)

    pyc_files = find_pyc_files(parent_root)

    print(f"找到 {len(pyc_files)} 个 .pyc 文件需要反编译")

    success = 0
    failed = 0

    for pyc_path, module_name, output_path in pyc_files:
        try:
            source = decompile_module(pyc_path, module_name)
            with open(output_path, 'w') as f:
                f.write(source)
            print(f"  [OK] {module_name} -> {output_path}")
            success += 1
        except Exception as e:
            print(f"  [FAIL] {module_name}: {e}")
            failed += 1

    print(f"\n反编译完成: {success} 成功, {failed} 失败")


if __name__ == '__main__':
    main()

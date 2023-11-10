#!/usr/bin/env python3

import unittest

from cpu import CPU, STACK, State, DATA, Memory, Flag
from cpu import Operand, RegisterOp, MemoryOp, ImmediateOp, OpType, Register


class CPUTest(unittest.TestCase):
    def test_init(self):
        sentinel = list()
        cpu = CPU(sentinel)

        self.assertEqual(STACK, cpu.sp)
        self.assertEqual(1, len(cpu.states))
        self.assertEqual(sentinel, cpu.instructions)
        self.assertEqual(None, cpu.data)

    def test_init_data(self):
        sentinel = list()
        cpu = CPU(sentinel, data=sentinel)

        self.assertEqual(STACK, cpu.sp)
        self.assertEqual(1, len(cpu.states))
        self.assertEqual(sentinel, cpu.instructions)
        self.assertEqual(sentinel, cpu.data)

    def test_load(self):
        cpu = CPU(list())

        data = {}
        for n, char in enumerate("hello"):
            data[n] = ord(char)
        cpu._load(data)

        self.assertEqual(None, cpu.data)
        self.assertEqual(cpu.memory[1 + DATA], data[1])
        self.assertEqual(5, len(cpu.memory))

    def test_rewrite_labels(self):
        program = [
                ('fake', [Operand(OpType.TEXTL, 1), MemoryOp(Register.SP)]),
                ('fake', [RegisterOp(Register.BP),  Operand(OpType.TEXTL, 1)]),
                ('fake', [Operand(OpType.TEXTL, 1), Operand(OpType.TEXTL, 1)]),
                ('fake', [RegisterOp(Register.BP),  MemoryOp(Register.SP)]),
                ('fake', [Operand(OpType.DATAL, 2), MemoryOp(Register.SP)]),
                ('fake', [RegisterOp(Register.BP),  Operand(OpType.DATAL, 2)]),
                ('fake', [Operand(OpType.DATAL, 2), Operand(OpType.DATAL, 2)]),
                ('fake', [Operand(OpType.DATAL, 2), Operand(OpType.TEXTL, 1)]),
                ('fake', [Operand(OpType.TEXTL, 1), Operand(OpType.DATAL, 2)]),
                ]

        rewrite = [
                ('fake', [ImmediateOp(0x5501),     MemoryOp(Register.SP)]),
                ('fake', [RegisterOp(Register.BP), ImmediateOp(0x5501)]),
                ('fake', [ImmediateOp(0x5501),     ImmediateOp(0x5501)]),
                ('fake', [RegisterOp(Register.BP), MemoryOp(Register.SP)]),
                ('fake', [ImmediateOp(0x6b02),     MemoryOp(Register.SP)]),
                ('fake', [RegisterOp(Register.BP), ImmediateOp(0x6b02)]),
                ('fake', [ImmediateOp(0x6b02),     ImmediateOp(0x6b02)]),
                ('fake', [ImmediateOp(0x6b02),     ImmediateOp(0x5501)]),
                ('fake', [ImmediateOp(0x5501),     ImmediateOp(0x6b02)])
                ]

        cpu = CPU([])
        self.maxDiff = None
        # self.assertEqual(rewrite, cpu._rewrite_labels(program))
        res = cpu._rewrite_labels(program)
        self.assertSequenceEqual(rewrite, res)

    def test_get_operand_value(self):
        state = State(registers={'ax': 0x22},
                      memory=Memory({0x22: 0x34, 0x23: 0x12}))
        cpu = CPU([], state=state)

        self.assertEqual(0x22, cpu._get_operand_value(RegisterOp(Register.AX)))

        self.assertEqual(0x1234, cpu._get_operand_value(MemoryOp(Register.AX)))
        self.assertEqual(0x0012, cpu._get_operand_value(MemoryOp(Register.AX, 1)))
        self.assertEqual(0x3400, cpu._get_operand_value(MemoryOp(Register.AX, -1)))

        self.assertEqual(0x22, cpu._get_operand_value(ImmediateOp(0x22)))

    def test_set_operand_value(self):
        pass

    def test_read_memory(self):
        pass

    def test_write_memory(self):
        pass

    def test_memory_operand(self):
        pass

    def test_op_mov(self):
        # confirm raises, test supported combinations
        cpu = CPU([])

        cpu.op_mov(RegisterOp(Register.AX), ImmediateOp(STACK))
        self.assertEqual(STACK, cpu.registers['ax'])

        cpu.op_mov(RegisterOp(Register.BX), RegisterOp(Register.AX))
        self.assertEqual(STACK, cpu.registers['bx'])

        cpu.op_mov(MemoryOp(Register.AX), RegisterOp(Register.AX))
        self.assertEqual(STACK, cpu._read_memory(STACK))

        with self.assertRaises(Exception):
            cpu.op_mov(MemoryOp(Register.AX), MemoryOp(Register.BX))

        with self.assertRaises(Exception):
            cpu.op_mov(ImmediateOp(0x1), RegisterOp(Register.AX))

        with self.assertRaises(Exception):
            state = State(registers={'ax': 0x1234, 'bx': 0x22})
            cpu = CPU([], state=state)
            cpu.op_mov(MemoryOp(Register.AX), RegisterOp(Register.BX))

    def test_op_mul(self):
        state = State(registers={'ax': 0x10, 'bx': 0x3, 'cx': 0x4},
                      memory=Memory({0x4: 0xef, 0x5: 0xbe}))

        # For fun, go-style tests:
        class Test:
            def __init__(self, multiplier, ax, dx, cf, of):
                self.multiplier = multiplier
                self.ax = ax
                self.dx = dx
                self.cf = cf
                self.of = of

            def __repr__(self):
                return str(self.__dict__)

        for test in (
                Test(RegisterOp(Register.BX), 0x30, 0, 0, 0),
                Test(MemoryOp(Register.CX), 0xeef0, 0xb, 1, 1),
                ):

            with self.subTest(test=test):
                cpu = CPU([], state=state.copy())
                cpu.op_mul(test.multiplier)
                self.assertEqual(test.ax, cpu.registers['ax'])
                self.assertEqual(test.dx, cpu.registers['dx'])
                if test.cf:
                    self.assertIn(Flag.CF, cpu.flags)
                    self.assertIn(Flag.OF, cpu.flags)
                else:
                    self.assertNotIn(Flag.CF, cpu.flags)
                    self.assertNotIn(Flag.OF, cpu.flags)

        with self.assertRaises(Exception):
            cpu = CPU([])
            cpu.op_mul((0x1, 'i'))  # invalid optype

    def test_op_div(self):
        state = State(registers={'ax': 0x10, 'bx': 0x3, 'cx': 0x4, 'dx': 0x3},
                      memory=Memory({0x4: 0xef, 0x5: 0xbe}))

        # For fun, go-style tests:
        class Test:
            def __init__(self, divisor, ax, dx, cf, of):
                self.divisor = divisor
                self.ax = ax
                self.dx = dx
                self.cf = cf
                self.of = of

            def __repr__(self):
                return str(self.__dict__)

        for test in (
                Test(RegisterOp(Register.BX), 0x5, 1, 0, 0),
                Test(MemoryOp(Register.CX), 0x4, 0x454, 0, 0),
                ):

            with self.subTest(test=test):
                cpu = CPU([], state=state.copy())
                cpu.op_div(test.divisor)
                self.assertEqual(test.ax, cpu.registers['ax'])
                self.assertEqual(test.dx, cpu.registers['dx'])
                if test.cf:
                    self.assertIn(Flag.CF, cpu.flags)
                else:
                    self.assertNotIn(Flag.CF, cpu.flags)

                if test.of:
                    self.assertIn(Flag.OF, cpu.flags)
                else:
                    self.assertNotIn(Flag.OF, cpu.flags)

        with self.assertRaises(Exception):
            cpu = CPU([])
            cpu.op_mul((0x1, 'i'))  # invalid optype

    def test_op_add(self):
        state = State(registers={'ax': 0x2, 'bx': 0x3, 'cx': 0x4},
                      memory=Memory({0x4: 0x34, 0x5: 0x12}))

        class Test:
            def __init__(self, dest, src, res):
                self.dest = dest
                self.src = src
                self.res = res

            def __repr__(self):
                return str(self.__dict__)

        for test in (
                Test(RegisterOp(Register.BX), ImmediateOp(0x2), 0x5),
                Test(RegisterOp(Register.AX), RegisterOp(Register.AX), 0x4),
                Test(RegisterOp(Register.BX), MemoryOp(Register.CX), 0x1237),
                ):

            with self.subTest(test=test):
                cpu = CPU([], state=state.copy())
                cpu.op_add(test.dest, test.src)
                self.assertEqual(test.res, cpu.registers[test.dest.value.value])

    def test_op_sub(self):
        state = State(registers={'ax': 0x2, 'bx': 0x2468, 'cx': 0x4},
                      memory=Memory({0x4: 0x34, 0x5: 0x12}))

        class Test:
            def __init__(self, dest, src, res):
                self.dest = dest
                self.src = src
                self.res = res

            def __repr__(self):
                return str(self.__dict__)

        for test in (
                Test(RegisterOp(Register.BX), ImmediateOp(0x2), 0x2466),
                Test(RegisterOp(Register.AX), RegisterOp(Register.AX), 0x0),
                Test(RegisterOp(Register.BX), RegisterOp(Register.AX), 0x2466),
                Test(RegisterOp(Register.BX), MemoryOp(Register.CX), 0x1234),
                ):

            with self.subTest(test=test):
                cpu = CPU([], state=state.copy())
                cpu.op_sub(test.dest, test.src)
                self.assertEqual(test.res,
                                 cpu.registers[test.dest.value.value])

    def test_flags(self):
        class Test:
            def __init__(self, arg1, arg2, res, of, sf, zf, cf):
                self.arg1 = arg1
                self.arg2 = arg2
                self.res = res
                self.of = of
                self.sf = sf
                self.zf = zf
                self.cf = cf

            def __repr__(self):
                res = {}
                for i in self.__dict__:
                    val = self.__dict__[i]
                    if isinstance(val, int) and not isinstance(val, bool):
                        res[i] = hex(val)
                    else:
                        res[i] = val
                return str(res)

            def checkflags(self, test, flags):
                if self.of:
                    test.assertIn(Flag.OF, flags)
                else:
                    test.assertNotIn(Flag.OF, flags)

                if self.sf:
                    test.assertIn(Flag.SF, flags)
                else:
                    test.assertNotIn(Flag.SF, flags)

                if self.zf:
                    test.assertIn(Flag.ZF, flags)
                else:
                    test.assertNotIn(Flag.ZF, flags)

                if self.cf:
                    test.assertIn(Flag.CF, flags)
                else:
                    test.assertNotIn(Flag.CF, flags)

        for test in (
                #    arg1     arg2     result   OF      SF      ZF      CF
                Test(0x7f00,  0,       0x7f00,  False,  False,  False,  False),
                Test(0xffff,  0x7f,    0x7e,    False,  False,  False,  True),
                Test(0,       0,       0,       False,  False,  True,   False),
                Test(0xffff,  0x1,     0,       False,  False,  True,   True),
                Test(0xffff,  0,       0xffff,  False,  True,   False,  False),
                Test(0xffff,  0xffff,  0xfffe,  False,  True,   False,  True),
                Test(0xffff,  0x8000,  0x7fff,  True,   False,  False,  True),
                Test(0x8000,  0x8000,  0,       True,   False,  True,   True),
                Test(0x7fff,  0x7fff,  0xfffe,  True,   True,   False,  False)
                ):
            with self.subTest(test=test):
                cpu = CPU([], state=State({'ax': test.arg1}))
                cpu.op_add(RegisterOp(Register.AX), ImmediateOp(test.arg2))
                self.assertEqual(test.res, cpu.registers[Register.AX.value])
                test.checkflags(self, cpu.flags)

        for test in (
                #    arg1     arg2     result   OF      SF      ZF      CF
                Test(0xffff,  0xfffe,  1,       False,  False,  False,  False),
                Test(0x7ffe,  0xffff,  0x7fff,  False,  False,  False,  True),
                Test(0xffff,  0xffff,  0,       False,  False,  True,   False),
                Test(0xffff,  0x7fff,  0x8000,  False,  True,   False,  False),
                Test(0xfffe,  0xffff,  0xffff,  False,  True,   False,  True),
                Test(0xfffe,  0x7fff,  0x7fff,  True,   False,  False,  False),
                Test(0x7fff,  0xffff,  0x8000,  True,   True,   False,  True),
                ):
            with self.subTest(test=test):
                cpu = CPU([], state=State({'ax': test.arg1}))
                cpu.op_sub(RegisterOp(Register.AX), ImmediateOp(test.arg2))
                self.assertEqual(test.res, cpu.registers[Register.AX.value])
                test.checkflags(self, cpu.flags)

    def test_op_cmp(self):
        state = State(registers={'ax': 0x2, 'bx': 0x1234, 'cx': 0x4},
                      memory=Memory({0x4: 0x34, 0x5: 0x12}))

        class Test:
            def __init__(self, dest, src, res):
                self.dest = dest
                self.src = src
                self.res = res

            def __repr__(self):
                return str(self.__dict__)

        for test in (
                Test(RegisterOp(Register.AX), ImmediateOp(0x2), 1),
                Test(RegisterOp(Register.BX), ImmediateOp(0x2), 0),
                Test(RegisterOp(Register.AX), RegisterOp(Register.BX), 0),
                Test(RegisterOp(Register.AX), RegisterOp(Register.AX), 1),
                Test(RegisterOp(Register.AX), MemoryOp(Register.CX), 0),
                Test(RegisterOp(Register.BX), MemoryOp(Register.CX), 1),
                ):

            with self.subTest(test=test):
                cpu = CPU([], state=state.copy())
                cpu.op_cmp(test.dest, test.src)
                if test.res:
                    self.assertIn(Flag.ZF, cpu.flags)
                else:
                    self.assertNotIn(Flag.ZF, cpu.flags)

        with self.assertRaises(Exception):
            cpu = CPU([], state=state.copy())
            cpu.op_cmp('bx', 'm'), ('ax', 'm')

        with self.assertRaises(Exception):
            cpu = CPU([], state=state.copy())
            cpu.op_cmp(1, 'i'), ('ax', 'm')

    def test_op_jmp(self):
        state = State(registers={'ax': 0x2, 'bx': 0x5517, 'cx': 0x4},
                      memory=Memory({0x4: 0x10, 0x5: 0x55}))

        class Test:
            def __init__(self, dest, ip):
                self.dest = dest
                self.ip = ip

            def __repr__(self):
                return str(self.__dict__)

        for test in (
                Test(RegisterOp(Register.BX), 0x5517),
                Test(MemoryOp(Register.CX), 0x5510)
                ):

            with self.subTest(test=test):
                cpu = CPU([], state=state.copy())
                cpu.op_jmp(test.dest)
                self.assertEqual(test.ip, cpu.ip)

    def test_op_jne(self):
        state = State(registers={'ax': 0x2, 'bx': 0x5517, 'cx': 0x4},
                      memory=Memory({0x4: 0x10, 0x5: 0x55}))

        class Test:
            def __init__(self, dest, zf, ip):
                self.dest = dest
                self.zf = zf
                self.ip = ip

            def __repr__(self):
                return str(self.__dict__)

        for test in (
                Test(RegisterOp(Register.BX), False, 0x5517),
                Test(RegisterOp(Register.BX), True, 0),
                Test(MemoryOp(Register.CX), False, 0x5510),
                Test(MemoryOp(Register.CX), True, 0)
                ):

            with self.subTest(test=test):
                cpu = CPU([], state=state.copy())
                if test.zf:
                    cpu.flags |= Flag.ZF
                cpu.op_jne(test.dest)
                self.assertEqual(test.ip, cpu.ip)

    def test_op_je(self):
        state = State(registers={'ax': 0x2, 'bx': 0x5517, 'cx': 0x4},
                      memory=Memory({0x4: 0x10, 0x5: 0x55}))

        class Test:
            def __init__(self, dest, zf, ip):
                self.dest = dest
                self.zf = zf
                self.ip = ip

            def __repr__(self):
                return str(self.__dict__)

        for test in (
                Test(RegisterOp(Register.BX), True, 0x5517),
                Test(RegisterOp(Register.BX), False, 0),
                Test(MemoryOp(Register.CX), True, 0x5510),
                Test(MemoryOp(Register.CX), False, 0)
                ):

            with self.subTest(test=test):
                cpu = CPU([], state=state.copy())
                if test.zf:
                    cpu.flags |= Flag.ZF
                cpu.op_je(test.dest)
                self.assertEqual(test.ip, cpu.ip)

    def test_op_push(self):
        state = State(registers={
            'ax': 0x2, 'bx': 0x5517, 'cx': 0x4, 'sp': STACK},
                      memory=Memory({0x4: 0x10, 0x5: 0x55}))

        class Test:
            def __init__(self, src, res):
                self.src = src
                self.res = res

            def __repr__(self):
                return f'push {self.src} expect {self.res}'

        for test in (
                Test(RegisterOp(Register.AX), 0x2),
                Test(MemoryOp(Register.CX), 0x5510),
                Test(ImmediateOp(0x1234), 0x1234),
                ):

            with self.subTest(test=test):
                cpu = CPU([], state=state.copy())
                sp = cpu.sp
                cpu.op_push(test.src)
                self.assertEqual(sp - 2, cpu.sp)
                self.assertEqual(test.res, cpu._read_memory(cpu.sp))

    def test_op_pop(self):
        pass

    def test_op_call(self):
        pass

    def test_op_ret(self):
        pass


class OperandTest(unittest.TestCase):
    def test_registerop(self):
        pass

    def test_memoryop(self):
        pass

    def test_immediateop(self):
        r = RegisterOp(Register.AX)
        self.assertIsInstance(r, RegisterOp)
        self.assertEqual(r.optype, OpType.REGISTER)
        self.assertEqual(r.value, Register.AX)

        with self.assertRaises(Exception):
            r = RegisterOp(None)

    def test_from_optype(self):
        o = Operand.from_optype(OpType.REGISTER, Register.AX)
        self.assertIsInstance(o, RegisterOp)
        self.assertEqual(o.optype, OpType.REGISTER)
        self.assertEqual(o.value, Register.AX)

        o = Operand.from_optype(OpType.MEMORY, Register.AX)
        self.assertIsInstance(o, MemoryOp)
        self.assertEqual(o.optype, OpType.MEMORY)
        self.assertEqual(o.value, Register.AX)

        o = Operand.from_optype(OpType.IMMEDIATE, 0x1)
        self.assertIsInstance(o, ImmediateOp)
        self.assertEqual(o.optype, OpType.IMMEDIATE)
        self.assertEqual(o.value, 0x1)

#!/usr/bin/env python3

import re

from collections import defaultdict
from collections.abc import MutableMapping
from enum import auto, Enum, Flag


STACK = 0x7f00
TEXT = 0x5500
DATA = 0x6b00


def parse_operand(op, text_labels=None, data_labels=None):
    op = op.strip()
    if op in text_labels:
        return Operand.from_optype(OpType.TEXTL, text_labels[op])
    elif op in data_labels:
        return Operand.from_optype(OpType.DATAL, data_labels[op])
    elif len(op) == 2 and op.isalpha():
        return Operand.from_optype(OpType.REGISTER, Register(op))
    elif op.startswith('[') and op.endswith(']'):
        return MemoryOp.from_str(op[1:-1])
    else:
        try:
            val = int(op, 16)
            if val < 0:
                raise ValueError('invalid immediate value')
            if val & 0xffff != val:
                raise Exception('immediate value out of range')
            return ImmediateOp(val)
        except ValueError:
            raise Exception('invalid operand "%s"' % op)

        # shouldn't be reachable
        raise Exception('unknown operand "%s"' % op)


def parse_operands(operands, text_labels=None, data_labels=None):
    if not text_labels:
        text_labels = tuple()
    if not data_labels:
        data_labels = tuple()

    ret = []
    for op in operands.split(','):
        ret.append(parse_operand(op, text_labels, data_labels))

    return ret


class OpType(Enum):
    MEMORY = auto()
    IMMEDIATE = auto()
    REGISTER = auto()
    DATAL = auto()
    TEXTL = auto()


class Register(Enum):
    AX = 'ax'
    AL = 'al'
    AH = 'ah'
    BX = 'bx'
    BL = 'bl'
    BH = 'bh'
    CX = 'cx'
    CL = 'cl'
    CH = 'ch'
    DX = 'dx'
    DL = 'dl'
    DH = 'dh'
    DI = 'di'
    SI = 'si'
    BP = 'bp'
    SP = 'sp'
    IP = 'ip'


class Flag(Flag):
    CF = auto()
    PF = auto()
    ZF = auto()
    SF = auto()
    OF = auto()


class Operand:
    def __init__(self, optype, value):
        if optype not in OpType:
            raise Exception('invalid optype %s' % optype)

        self.optype = optype
        self.value = value

    def __eq__(self, other):
        return (isinstance(other, Operand)
                and self.optype == other.optype
                and self.value == other.value)

    def __repr__(self):
        return f'Operand({self.optype}, {self.value})'

    @classmethod
    def from_optype(cls, optype, *args, **kwargs):
        if optype == OpType.REGISTER:
            return RegisterOp(*args, **kwargs)
        elif optype == OpType.MEMORY:
            return MemoryOp(*args, **kwargs)
        elif optype == OpType.IMMEDIATE:
            return ImmediateOp(*args, **kwargs)
        else:
            return Operand(optype, *args, **kwargs)


class RegisterOp(Operand):
    def __init__(self, register):
        if register not in Register:
            raise Exception('invalid register %s' % register)
        super().__init__(OpType.REGISTER, register)


class MemoryOp(Operand):
    def __init__(self, base=None, index=None, scale=1, disp=0):
        super().__init__(OpType.MEMORY, base or index)
        self.base = base
        self.index = index
        self.disp = disp
        self.scale = scale

    def __eq__(self, other):
        return (isinstance(other, MemoryOp)
                and self.__dict__ == other.__dict__)

    @classmethod
    def from_str(cls, tokens):
        patterns = [
            # ( base | disp )
            r'((?P<disp>0x[0-9a-f]+)|(?P<base>[a-z]{2}))$',

            # base ( index | disp )
            (
                r'(?P<base>[a-z]{2})'
                r'(\+((?P<index>[a-z]{2}))|(?P<disp>(\+|-)0x[0-9a-f]+))$'
            ),

            # base index disp
            (
                r'(?P<base>[a-z]{2})'
                r'\+(?P<index>[a-z]{2})(?P<disp>(\+|-)0x[0-9a-f]+)$'
            ),

            # [ base ] index * scale [ disp ]
            (
                r'((?P<base>[a-z]{2})\+)?'
                r'(?P<index>[a-z]{2})\*(?P<scale>0x[124])'
                r'(?P<disp>(\+|-)0x[0-9a-f]+)?$'
            ),
            ]
        for pattern in patterns:
            match = re.compile(pattern).match(tokens)
            if match:
                break
        else:
            raise Exception('no matching format pattern for %s', tokens)

        res = match.groupdict()
        if 'disp' in res and res['disp']:
            res['disp'] = int(res['disp'], 16)

        if 'scale' in res and res['scale']:
            res['scale'] = int(res['scale'], 16)

        if 'base' in res and res['base']:
            res['base'] = Register(res['base'])

        if 'index' in res and res['index']:
            res['index'] = Register(res['index'])

        res = {x: y for x, y in res.items() if y is not None}
        return cls(**res)

    def get(self, cpu):
        addr = 0
        if self.index:
            addr += (cpu.registers[self.index.value] * self.scale)
        if self.base:
            addr += cpu.registers[self.base.value]
        if self.disp:
            addr += self.disp

        return addr


class ImmediateOp(Operand):
    def __init__(self, address):
        super().__init__(OpType.IMMEDIATE, address)


class Memory(MutableMapping):
    def __init__(self, memory=None):
        self.memory = defaultdict(int)
        if memory:
            self.update(memory)

    def copy(self):
        return Memory(self.memory)

    def __getitem__(self, key):
        return self.memory.__getitem__(key)

    def __delitem__(self, key):
        return self.memory.__delitem__(key)

    def __setitem__(self, key, value):
        return self.memory.__setitem__(key, value)

    def __iter__(self):
        return self.memory.__iter__()

    def __len__(self):
        return self.memory.__len__()


class State:
    def __init__(self, registers=None, flags=None, memory=None):
        self.registers = {reg: 0 for reg in CPU._registers}
        if registers:
            self.registers.update(registers)

        self.flags = Flag(0)
        if flags:
            self.flags |= flags

        if memory:
            self.memory = memory
        else:
            self.memory = Memory()

    def __repr__(self):
        return f'State({self.registers=}, {self.flags=}, {self.memory=})'

    def copy(self):
        return State(self.registers, self.flags, self.memory)


def _twos_complement(val):
    return val & 0xffff


def __twos_complement(val):
    bits = 16
    if (val & (1 << (bits - 1))) != 0:
        val = val - (1 << bits)

    return val & ((1 << bits) - 1)


def _twos_uncomplement(val):
    bits = 16
    if (val & (1 << (bits - 1))) != 0:
        val = -1 * ((1 << bits) - val)

    return val


class CPU:
    _registers = ['ax', 'bx', 'cx', 'dx', 'si', 'di', 'ip', 'bp', 'sp']

    def __init__(self, program, offset=0, data=None, state=None):
        if data and state:
            raise Exception('initial data and state cannot both be provided')

        self.instructions = self._rewrite_labels(program.copy())
        self.data = data  # to detect if data is present

        if state:
            self.states = [state.copy()]
        else:
            self.states = [State()]
            self.states[-1].registers['sp'] = STACK
            self.states[-1].registers['bp'] = 0x1
            self.states[-1].registers['ip'] = offset + TEXT
            if data:
                self._load(data)

    def _load(self, data):
        for addr in data:
            self.memory[addr + DATA] = data[addr] & 0xffff

    def _rewrite_labels(self, instructions):
        newi = []
        for i in range(len(instructions)):
            op, operands = instructions[i]
            new_operands = []
            for operand in operands:
                if operand.optype == OpType.TEXTL:
                    operand = Operand.from_optype(OpType.IMMEDIATE,
                                                  operand.value + TEXT)
                elif operand.optype == OpType.DATAL:
                    operand = Operand.from_optype(OpType.IMMEDIATE,
                                                  operand.value + DATA)
                new_operands.append(operand)
            newi.append((op, new_operands))

        return newi

    def prev(self):
        if len(self.states) > 1:
            self.states.pop()

    def _get_operand_value(self, operand):
        if operand.optype == OpType.REGISTER:
            register = operand.value.value
            if register.endswith('l'):
                # Low byte of register
                register = Register(register.replace('l', 'x'))
                return self.registers[register.value] & 0xff
            elif register.endswith('h'):
                # High byte of register
                register = Register(register.replace('h', 'x'))
                return (self.registers[register.value] & 0xff00) >> 8
            else:
                # Word size in register
                return self.registers[operand.value.value]
        elif operand.optype == OpType.MEMORY:
            addr = operand.get(self)
            return self._read_memory(addr)
        elif operand.optype == OpType.IMMEDIATE:
            return operand.value
        else:
            raise Exception('unknown operand type %s' % operand.optype)

    def _set_operand_value(self, operand, value):
        if operand.optype == OpType.REGISTER:
            register = operand.value.value
            if register.endswith('l'):
                # Low byte of register
                register = Register(register.replace('l', 'x'))
                value = value & 0xff
                cur = self.registers[register.value]
                self.registers[register.value] = (cur & 0xff00) | value
            elif register.endswith('h'):
                # High byte of register
                register = Register(register.replace('h', 'x'))
                value = value & 0xff
                cur = self.registers[register.value]
                self.registers[register.value] = (cur & 0xff) | (value << 8)
            else:
                # Word size in register
                self.registers[operand.value.value] = value & 0xffff
        elif operand.optype == OpType.MEMORY:
            # todo
            addr = operand.get(self)
            self._write_memory(addr, value)
        else:
            raise Exception('unknown operand type %s' % operand)

    def _read_memory(self, addr):
        b1, b2 = self.memory[addr], self.memory[addr+1]
        return (b2 << 8) | b1

    def _write_memory(self, addr, value):
        value = value & 0xffff
        self.memory[addr] = value & 0xff    # lower byte
        self.memory[addr+1] = value >> 0x8  # upper byte

    def op_ret(self):
        self.op_jmp(MemoryOp(Register.SP))
        self.op_add(RegisterOp(Register.SP), ImmediateOp(0x02))

    def op_call(self, operand):
        self.op_push(RegisterOp(Register.IP))
        self.op_jmp(operand)

    def op_push(self, source):
        self.op_sub(RegisterOp(Register.SP), ImmediateOp(0x02))
        self.op_mov(MemoryOp(Register.SP), source, force=True)

    def op_pop(self, dest):
        self.op_mov(dest, MemoryOp(Register.SP))
        self.op_add(RegisterOp(Register.SP), ImmediateOp(0x02))

    def op_jmp(self, dest):
        addr = self._get_operand_value(dest)

        if addr & 0xff00 != TEXT:
            raise Exception('invalid jmp target %x' % addr)

        self.registers['ip'] = addr

    def op_jne(self, operand):
        if Flag.ZF not in self.flags:
            self.op_jmp(operand)

    def op_je(self, operand):
        if Flag.ZF in self.flags:
            self.op_jmp(operand)

    def op_ja(self, operand):
        if Flag.CF not in self.flags and Flag.OF not in self.flags:
            self.op_jmp(operand)

    def op_jae(self, operand):
        if Flag.CF not in self.flags:
            self.op_jmp(operand)

    def op_jb(self, operand):
        if Flag.CF in self.flags:
            self.op_jmp(operand)

    def op_jbe(self, operand):
        if Flag.CF in self.flags or Flag.ZF in self.flags:
            self.op_jmp(operand)

    def op_jc(self, operand):
        if Flag.CF in self.flags:
            self.op_jmp(operand)

    def op_jg(self, operand):
        if Flag.ZF not in self.flags:
            if Flag.SF in self.flags and Flag.OF in self.flags:
                self.op_jmp(operand)
            elif Flag.SF not in self.flags and Flag.OF not in self.flags:
                self.op_jmp(operand)

    def op_jge(self, operand):
        if Flag.SF in self.flags and Flag.OF in self.flags:
            self.op_jmp(operand)
        elif Flag.SF not in self.flags and Flag.OF not in self.flags:
            self.op_jmp(operand)

    def op_jl(self, operand):
        if Flag.SF in self.flags and Flag.OF not in self.flags:
            self.op_jmp(operand)
        elif Flag.SF not in self.flags and Flag.OF in self.flags:
            self.op_jmp(operand)

    def op_jle(self, operand):
        if Flag.SF in self.flags and Flag.OF not in self.flags:
            self.op_jmp(operand)
        elif Flag.SF not in self.flags and Flag.OF in self.flags:
            self.op_jmp(operand)
        elif Flag.ZF in self.flags:
            self.op_jmp(operand)

# Reduce convenient combinations; think about how to use flags to check values.
#
# JNA  Jump short if not above (CF=1 or ZF=1).
# JNAE Jump short if not above or equal (CF=1).
# JNB  Jump short if not below (CF=0).
# JNBE Jump short if not below or equal (CF=0 and ZF=0).
# JNC  Jump short if not carry (CF=0).
# JNE  Jump short if not equal (ZF=0).
# JNG  Jump short if not greater (ZF=1 or SF≠ OF).
# JNGE Jump short if not greater or equal (SF≠ OF).
# JNL  Jump short if not less (SF=OF).
# JNLE Jump short if not less or equal (ZF=0 and SF=OF).
# JNO  Jump short if not overflow (OF=0).
# ###JNP  Jump short if not parity (PF=0).
# JNS  Jump short if not sign (SF=0).
# JNZ  Jump short if not zero (ZF=0).
# JO   Jump short if overflow (OF=1).
# ###JP   Jump short if parity (PF=1).
# ###JPE  Jump short if parity even (PF=1).
# ###JPO  Jump short if parity odd (PF=0).
# JS   Jump short if sign (SF=1).
# JZ   Jump short if zero (ZF = 1).

    def op_cmp(self, dest, src):
        if src.optype == dest.optype == OpType.MEMORY:
            raise Exception('invalid source,dest pair (%s,%s)' % (dest, src))
        if dest.optype == OpType.IMMEDIATE:
            raise Exception('invalid dest operand')

        srcval = self._get_operand_value(src)
        destval = self._get_operand_value(dest)

        if destval - srcval == 0:
            self.flags |= Flag.ZF
        else:
            self.flags &= ~Flag.ZF

    def op_sub(self, dest, src):
        if src.optype == dest.optype == OpType.MEMORY:
            raise Exception('invalid source,dest pair (%s,%s)' % (dest, src))
        if dest.optype == OpType.IMMEDIATE:
            raise Exception('invalid dest operand')

        destval = self._get_operand_value(dest)
        srcval = self._get_operand_value(src)
        res = destval - srcval

        if res < 0:
            res = _twos_complement(res)
            self.flags |= Flag.CF
        else:
            self.flags &= ~Flag.CF

        if res == 0:
            self.flags |= Flag.ZF
        else:
            self.flags &= ~Flag.ZF

        if res & 0x8000 == 0x8000:
            self.flags |= Flag.SF
        else:
            self.flags &= ~Flag.SF

        if (
                srcval & 0x8000 == 0x8000
                and destval & 0x8000 == 0
                and res & 0x8000 == 0x8000):
            self.flags |= Flag.OF
        elif (
                srcval & 0x8000 == 0
                and destval & 0x8000 == 0x8000
                and res & 0x8000 == 0):
            self.flags |= Flag.OF
        else:
            self.flags &= ~Flag.OF

        self._set_operand_value(dest, res)

    def op_add(self, dest, src):
        if src.optype == dest.optype == OpType.MEMORY:
            raise Exception('invalid source,dest pair (%s,%s)' % (dest, src))
        if dest.optype == OpType.IMMEDIATE:
            raise Exception('invalid dest operand')

        srcval = self._get_operand_value(src)
        destval = self._get_operand_value(dest)
        res = destval + srcval

        if res > res & 0xffff:
            res = res & 0xffff
            self.flags |= Flag.CF
        else:
            self.flags &= ~Flag.CF

        if res == 0:
            self.flags |= Flag.ZF
        else:
            self.flags &= ~Flag.ZF

        if res & 0x8000 == 0x8000:
            self.flags |= Flag.SF
        else:
            self.flags &= ~Flag.SF

        if (
                srcval & 0x8000 == 0x8000
                and destval & 0x8000 == 0x8000
                and res & 0x8000 == 0):
            self.flags |= Flag.OF
        elif (
                srcval & 0x8000 == 0
                and destval & 0x8000 == 0
                and res & 0x8000 == 0x8000):
            self.flags |= Flag.OF
        else:
            self.flags &= ~Flag.OF

        self._set_operand_value(dest, res)

    def op_mul(self, operand):
        if operand.optype == OpType.IMMEDIATE:
            raise Exception('invalid operand "%s"' % operand)

        multiplier = self._get_operand_value(operand)
        product = (multiplier * self.registers['ax']) & 0xffffffff
        self.registers['dx'] = product >> 16     # high word
        self.registers['ax'] = product & 0xffff  # low word

        if self.registers['dx'] == 0:
            self.flags &= ~Flag.CF
            self.flags &= ~Flag.OF
        else:
            self.flags |= Flag.CF
            self.flags |= Flag.OF

    def op_imul(self, operand):
        # typically we would need sign extension prior to binary multiplication
        # but we can just decode two's complement, multiply, convert back.
        if operand.optype == OpType.IMMEDIATE:
            raise Exception('invalid operand "%s"' % operand)

        multiplier = self._get_operand_value(operand)
        multiplier = _twos_uncomplement(multiplier)
        multiplicand = _twos_uncomplement(self.registers['ax'])
        # TODO product = _twos_complement(multiplier * multiplicand, 32)
        product = multiplier * multiplicand
        self.registers['dx'] = _twos_complement(product >> 16)     # high word
        self.registers['ax'] = product & 0xffff                    # low word

        if self.registers['dx'] == 0:
            self.flags &= ~Flag.CF
            self.flags &= ~Flag.OF
        else:
            self.flags |= Flag.CF
            self.flags |= Flag.OF

    def op_div(self, operand):
        if operand.optype == OpType.IMMEDIATE:
            raise Exception('invalid operand "%s"' % operand)

        divisor = self._get_operand_value(operand)

        dividend = self.registers['ax'] & 0xffff
        dividend |= self.registers['dx'] << 16

        quotient = int(dividend / divisor)
        if quotient > 0xFFFFFFFF:
            raise Exception("divide error")
        self._set_operand_value(RegisterOp(Register.AX), quotient)

        remainder = dividend % divisor
        self._set_operand_value(RegisterOp(Register.DX), remainder)

    def op_idiv(self, operand):
        if operand.optype == OpType.IMMEDIATE:
            raise Exception('invalid operand "%s"' % operand)

        divisor = self._get_operand_value(operand)
        divisor = _twos_uncomplement(divisor)
        dividend = _twos_uncomplement(self.registers['ax'])
        dividend |= self.registers['dx'] << 16
        # dividend = _twos_uncomplement(dividend)
        # TODO: dividend = _twos_uncomplement(dividend, 32)

        quotient = int(dividend / divisor)
        if quotient > 0xFFFFFFFF:
            raise Exception("divide error")

        self._set_operand_value(RegisterOp(Register.AX),
                                _twos_complement(quotient))

        if dividend < 0 or divisor < 0:
            remainder = (-1 * dividend) % divisor
            remainder = -1 * remainder
        else:
            remainder = dividend % divisor
        self._set_operand_value(RegisterOp(Register.DX),
                                _twos_complement(remainder))

    def op_mov(self, dest, src, force=False):
        if src.optype == dest.optype == OpType.MEMORY:
            if not force:
                raise Exception('invalid source,dest pair (%s)'
                                % tuple([dest, src]))

        if dest.optype == OpType.IMMEDIATE:
            raise Exception('invalid dest operand')

        for op in (src, dest):
            if op.optype == OpType.REGISTER:
                if op.value.value[1] in ('h', 'l'):
                    raise Exception('invalid operand size "%s"' % op)

        if dest.optype == OpType.MEMORY:
            # TODO: move to write_mem
            addr = dest.get(self)
            if addr & 0xf000 != 0x7000:
                raise Exception(
                        f'runtime error: invalid memory address {addr:#06x}')

        opval = self._get_operand_value(src)
        self._set_operand_value(dest, opval)

    def op_movb(self, dest, src):
        if src.optype == dest.optype == OpType.MEMORY:
            raise Exception('invalid source,dest pair (%s)'
                            % tuple([dest, src]))

        if dest.optype == OpType.IMMEDIATE:
            raise Exception('invalid dest operand')

        for op in (src, dest):
            if op.optype == OpType.REGISTER:
                if op.value.value[1] in ('h', 'l'):
                    raise Exception('invalid operand size "%s"' % op)

        if dest.optype == OpType.MEMORY:
            # TODO: move to write_mem
            addr = dest.get(self)
            if addr & 0xf000 != 0x7000:
                raise Exception(
                        f'runtime error: invalid memory address {addr:#06x}')

        opval = self._get_operand_value(src)
        self._set_operand_value(dest, opval)

    def op_shl(self, dest, count):
        if count.optype == OpType.REGISTER and count.value != Register.CL:
            raise Exception('invalid operand "%s"' % count)
        if dest.optype == OpType.IMMEDIATE:
            raise Exception('invalid operand "%s"' % dest)

        curval = self._get_operand_value(dest)
        countval = self._get_operand_value(count)
        self._set_operand_value(dest, curval << countval)

    def op_shlb(self, dest, count):
        if count.optype == OpType.REGISTER and count.value != Register.CL:
            raise Exception('invalid operand "%s"' % count)
        if dest.optype == OpType.IMMEDIATE:
            raise Exception('invalid operand "%s"' % dest)
        if dest.optype == OpType.REGISTER:
            if dest.value.value[1] not in ('l', 'h'):
                raise Exception('invalid operand "%s"' % dest)

        destval = self._get_operand_value(dest)
        self._set_operand_value(dest, destval << count)

    def op_shr(self, dest, count):
        if count.optype == OpType.REGISTER and count.value != Register.CL:
            raise Exception('invalid operand "%s"' % count)
        if dest.optype == OpType.IMMEDIATE:
            raise Exception('invalid operand "%s"' % dest)

        curval = self._get_operand_value(dest)
        countval = self._get_operand_value(count)
        self._set_operand_value(dest, curval >> countval)

    def op_sar(self, dest, count):
        if count.optype == OpType.REGISTER and count.value != Register.CL:
            raise Exception('invalid operand "%s"' % count)
        if dest.optype == OpType.IMMEDIATE:
            raise Exception('invalid operand "%s"' % dest)

        curval = self._get_operand_value(dest)
        countval = self._get_operand_value(count)
        res = curval >> countval
        if (curval & 0x8000):
            mask = ((1 << countval) - 1) << (16 - countval)
            res = res | mask
        self._set_operand_value(dest, res)

    def op_xor(self, dest, src):
        if src.optype == dest.optype == OpType.MEMORY:
            raise Exception('invalid source,dest pair (%s,%s)' % (dest, src))
        if dest.optype == OpType.IMMEDIATE:
            raise Exception('invalid dest operand')
        if dest.optype == OpType.REGISTER:
            if dest.value.value[1] in ('l', 'h'):
                raise Exception('invalid operand "%s"' % dest)
        if src.optype == OpType.REGISTER:
            if src.value.value[1] in ('l', 'h'):
                raise Exception('invalid operand "%s"' % src)

        srcval = self._get_operand_value(src)
        destval = self._get_operand_value(dest)
        res = destval ^ srcval

        self.flags &= ~Flag.CF
        self.flags &= ~Flag.OF

        if res == 0:
            self.flags |= Flag.ZF
        else:
            self.flags &= ~Flag.ZF

        if res & 0x8000 == 0x8000:
            self.flags |= Flag.SF
        else:
            self.flags &= ~Flag.SF

        self._set_operand_value(dest, res)

    def op_not(self, dest):
        if dest.optype == OpType.IMMEDIATE:
            raise Exception('invalid operand "%s"' % dest)

        destval = self._get_operand_value(dest)
        res = (0xffff - destval) & 0xffff
        self._set_operand_value(dest, res)

    def op_or(self, dest, src):
        if src.optype == dest.optype == OpType.MEMORY:
            raise Exception('invalid source,dest pair (%s,%s)' % (dest, src))
        if dest.optype == OpType.IMMEDIATE:
            raise Exception('invalid dest operand')

        destval = self._get_operand_value(dest)
        srcval = self._get_operand_value(src)
        res = (destval | srcval) & 0xffff

        self.flags &= ~Flag.CF
        self.flags &= ~Flag.OF

        if res == 0:
            self.flags |= Flag.ZF
        else:
            self.flags &= ~Flag.ZF

        if res & 0x8000 == 0x8000:
            self.flags |= Flag.SF
        else:
            self.flags &= ~Flag.SF

        self._set_operand_value(dest, res)

    def op_and(self, dest, src):
        if src.optype == dest.optype == OpType.MEMORY:
            raise Exception('invalid source,dest pair (%s,%s)' % (dest, src))
        if dest.optype == OpType.IMMEDIATE:
            raise Exception('invalid dest operand')

        destval = self._get_operand_value(dest)
        srcval = self._get_operand_value(src)
        res = (destval & srcval) & 0xffff

        self.flags &= ~Flag.CF
        self.flags &= ~Flag.OF

        if res == 0:
            self.flags |= Flag.ZF
        else:
            self.flags &= ~Flag.ZF

        if res & 0x8000 == 0x8000:
            self.flags |= Flag.SF
        else:
            self.flags &= ~Flag.SF

        self._set_operand_value(dest, res)

    def op_hlt(self):
        # TODO: handle this and simply display/stop taking input
        raise Exception('halt!')

    def op_nop(self):
        return

    def execute(self):
        self.states.append(State(self.registers.copy(), self.flags,
                                 self.memory.copy()))

        ip = self.ip
        self.registers['ip'] += 1
        op, operands = self.instructions[ip - TEXT]
        op = 'op_' + op
        opfunc = getattr(self, op, None)
        if opfunc:
            opfunc(*operands)
        else:
            raise Exception('unsupported operation "%s"' % op)

    @property
    def registers(self):
        return self.states[-1].registers

    @property
    def flags(self):
        return self.states[-1].flags

    @flags.setter
    def flags(self, value):
        self.states[-1].flags = value

    @property
    def memory(self):
        return self.states[-1].memory

    @property
    def ip(self):
        return self.registers['ip']

    @property
    def sp(self):
        return self.registers['sp']

#!/usr/bin/env python3

from cpu import DATA, STACK, TEXT, Flag

import curses
import curses.panel


COLOR_NORMAL = 0
COLOR_RED = 1
COLOR_BLUE = 2
COLOR_YELLOW = 3


class Window:
    def __init__(self, title, nlines, ncols, begin_y, begin_x):
        self.title = title

        self.frame = curses.newwin(0, 0)
        self.frame_panel = curses.panel.new_panel(self.frame)

        self.w = curses.newwin(0, 0)
        self.w_panel = curses.panel.new_panel(self.w)

        self.selected = None
        self.items = []
        self.active = False
        self.start = 0

        self.resize(nlines, ncols, begin_y, begin_x)

    def resize(self, n_lines, n_cols, start_y, start_x):
        self.frame.resize(n_lines, n_cols)
        self.frame_panel.move(start_y, start_x)
        self.frame.erase()
        self.frame.addstr(1, 2, self.title, curses.color_pair(1))

        self.w.resize(n_lines - 6, n_cols - 4)
        self.w_panel.move(start_y + 4, start_x + 2)

        self.refresh()

    def refresh(self, update_start=False):
        if self.active:
            self.frame.attron(curses.color_pair(COLOR_RED))
            self.frame.box()
            self.frame.attrset(0)
        else:
            self.frame.box()
        self.frame.noutrefresh()

        self.w.erase()
        max_y, max_x = self.w.getmaxyx()

        # TODO: don't allow scrolling past end of items
        if update_start and self.selected is not None:
            if self.selected >= (self.start + max_y):
                self.start = self.selected - max_y + 1
            elif self.selected <= self.start:
                self.start = self.selected

        if self.start < 0:
            self.start = 0
        elif self.start > len(self.items):
            self.start = len(self.items) - 1

        for n, item in enumerate(self.items[self.start:]):
            if n >= max_y:
                break
            self.drawcolorline(n, 0, item)

        if self.selected is None:
            return

        if self.selected < self.start:
            return

        if self.selected >= (self.start + max_y):
            return

        self.w.chgat(self.selected - self.start, 0, -1,
                     curses.A_REVERSE | curses.A_BOLD)

    def drawcolorline(self, y, x, line):
        self.w.move(y, x)
        for color, split in line:
            if color:
                attr = curses.color_pair(color)
            else:
                attr = curses.A_NORMAL
            self.w.addstr(split, attr)

    def setitems(self, items, refresh=True):
        self.items = items
        if refresh:
            self.refresh()

    def select(self, sel):
        if sel > len(self.items) or sel < 0:
            return
            # raise Exception('index out of range')
        self.selected = sel


class RegisterWindow(Window):
    def update(self, cpu):
        registers = []
        for n, reg in enumerate(cpu.registers):
            registers.append([(COLOR_BLUE, f'{reg}'),
                              (COLOR_NORMAL, f' {cpu.registers[reg]:#06x}')])

        flagline = f'flags {cpu.flags.value:#x}'
        for flag in Flag:
            if flag in cpu.flags:
                name = flag.name
            else:
                name = '  '
            flagline = f'{flagline} {name}'

        registers.append([(COLOR_NORMAL, flagline)])

        self.setitems(registers)


class MemoryWindow(Window):
    def update(self, cpu):
        cpu_memory = cpu.memory.copy()

        start = DATA if cpu.data else STACK - 0x100
        end = STACK+0x100

        memory = []
        for n, addr in enumerate(range(start, end, 4)):
            dataline = ''
            for i in range(4):
                data = cpu_memory[addr+i]
                dataline = f'{dataline} {data:#04x}'
            line = [(COLOR_BLUE, f'{addr:#06x}'),
                    (COLOR_NORMAL, dataline)]
            if addr == cpu.sp or addr + 2 == cpu.sp:
                line.append((COLOR_NORMAL, '  <-'))
                if self.start is None:
                    self.start = n

            memory.append(line)

        self.setitems(memory)


class TextWindow(Window):
    def update(self, cpu, program, follow=False):
        formatted = []
        for addr, i in enumerate(program):
            formatted.append([(COLOR_BLUE, f'{addr + TEXT:#06x}'),
                              (COLOR_NORMAL, f' {i}')])
        for label, addr in self.labels.items():
            formatted[addr].extend([(COLOR_YELLOW, f' # {label}')])
        self.setitems(formatted, False)
        self.select(cpu.ip - TEXT)
        self.refresh(follow)

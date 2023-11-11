#!/usr/bin/env python3

from x86cpu.cpu import CPU, TEXT, parse_operands
from x86cpu.window import MemoryWindow, RegisterWindow, TextWindow
from x86cpu.window import COLOR_RED, COLOR_BLUE, COLOR_YELLOW

import curses
import curses.panel
import sys
import traceback

from collections import defaultdict
from curses import wrapper


WIN_REGISTER_WIDTH = 15
WIN_TEXT_WIDTH = 35
WIN_HEIGHT = 13


def parse_program(program):
    labels = {'text': {}, 'data': {}}
    sections = {'text': [], 'data': []}
    section = 'text'

    n = 0
    for line in program:
        line = line.split(';')[0].strip()
        if not line:
            continue

        if line.endswith(':'):
            labels[section][line[:-1]] = n
        elif line.startswith('.section'):
            section = line.split()[1]
            n = 0
        else:
            sections[section].append(line)
            n += 1

    if '_start' not in labels['text']:
        labels['text']['_start'] = 0

    return sections, labels


def parse_text(text, labels):
    instructions = []
    for n, instruction in enumerate(text):
        instruction = instruction.split(maxsplit=1)
        if len(instruction) < 2:
            instruction = instruction[0]
            if instruction not in ['nop', 'ret', 'hlt']:
                raise Exception('invalid instruction: %s requires operands'
                                % instruction)
            instructions.append((instruction, tuple()))
        else:
            op, operands = instruction
            operands = parse_operands(operands, labels['text'], labels['data'])
            instructions.append((op, operands))

    return instructions


def parse_data(data):
    new_data = defaultdict(int)
    for n, instruction in enumerate(data.copy()):
        op, operands = instruction.split(maxsplit=1)
        if op == '.string':
            if not (operands.startswith('"') and operands.endswith('"')):
                raise Exception('invalid string value "%s"' % instruction)
            operands = operands[1:-1]
            for i, char in enumerate(operands):
                new_data[n+i] = ord(char)
        else:
            raise Exception('invalid instruction "%s"' % instruction)

    return new_data


def _main(stdscr):
    global debug

    err_win_panel = None

    stdscr.clear()

    curses.init_pair(COLOR_RED, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(COLOR_BLUE, curses.COLOR_BLUE, curses.COLOR_BLACK)
    curses.init_pair(COLOR_YELLOW, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.curs_set(0)  # hide cursor

    minwidth = WIN_REGISTER_WIDTH + WIN_TEXT_WIDTH
    minheight = WIN_HEIGHT * 2

    if curses.COLS < minwidth or curses.LINES < minheight:
        raise Exception('window too small - %dx%d is less than minimum %dx%d'
                        % (curses.COLS, curses.LINES, minwidth, minheight))

    max_y, max_x = stdscr.getmaxyx()
    lower_win_height = max(WIN_HEIGHT, max_y // 3)

    fn = sys.argv[1] if len(sys.argv) > 1 else 'asm.txt'
    with open(fn, 'r') as fh:
        lines = fh.read().splitlines()

    # TODO: desperate need of renaming
    sections, labels = parse_program(lines)
    program = parse_text(sections['text'], labels)
    data = parse_data(sections['data'])

    cpu = CPU(program, labels['text']['_start'], data)

    # Memory window: entire lower region
    memory_win = MemoryWindow(
            'Stack',
            lower_win_height,
            curses.COLS,
            curses.LINES - lower_win_height,
            0)
    memory_win.start = None

    upper_win_width = max(WIN_REGISTER_WIDTH, max_x // 3)

    # Text window: left 2/3rds of upper region.
    text_win = TextWindow(
            'Text',
            curses.LINES - lower_win_height,
            curses.COLS - upper_win_width,
            0,
            0)

    text_win.labels = labels['text']

    # Register window: right 1/3rd of upper region.
    register_win = RegisterWindow(
            'Registers',
            curses.LINES - lower_win_height,
            curses.COLS - (curses.COLS - upper_win_width),
            0,
            curses.COLS - upper_win_width)

    refresh = True
    parse_mode = False
    windows = [text_win, memory_win]
    selwin = 0
    follow = True
    while True:
        if refresh:
            windows[selwin].active = True
            windows[selwin-1].active = False

            text_win.update(cpu, program if parse_mode else sections['text'],
                            follow)
            memory_win.update(cpu)
            register_win.update(cpu)
            refresh = False

        curses.panel.update_panels()
        curses.doupdate()

        inp = stdscr.getch()
        if inp == curses.KEY_UP or inp == ord('k'):
            if windows[selwin] != text_win:
                continue
            if err_win_panel is not None:
                continue
            cpu.prev()
            follow = True
            refresh = True

        elif inp == curses.KEY_DOWN or inp == ord('j'):
            if windows[selwin] != text_win:
                continue
            if err_win_panel is not None:
                continue
            if cpu.ip + 1 == len(cpu.instructions) + TEXT:
                continue
            cpu.execute()
            follow = True
            refresh = True

        elif inp == ord('J'):
            if err_win_panel is not None:
                continue
            if windows[selwin].start >= len(windows[selwin].items):
                continue
            windows[selwin].start += 1
            follow = False
            refresh = True

        elif inp == ord('K'):
            if err_win_panel is not None:
                continue
            if windows[selwin].start <= 0:
                continue
            windows[selwin].start -= 1
            follow = False
            refresh = True

        elif inp == ord('0'):
            windows[selwin].start = 0
            refresh = True

        elif inp == ord('1'):
            windows[selwin].start = len(windows[selwin].items) - 1
            refresh = True

        elif inp == ord('\t'):
            # Cycle active window
            if selwin:
                selwin = 0
            else:
                selwin = 1
            refresh = True

        elif inp == curses.KEY_RESIZE:
            stdscr.erase()
            stdscr.noutrefresh()

            max_y, max_x = stdscr.getmaxyx()
            lower_win_height = max(WIN_HEIGHT, max_y // 3)
            upper_win_width = max(WIN_REGISTER_WIDTH, max_x // 3)

            if (
                    max_y < (2 * WIN_HEIGHT)
                    or max_x < (WIN_TEXT_WIDTH + WIN_REGISTER_WIDTH)):
                if not err_win_panel:
                    err_win = curses.newwin(0, 0)
                    err_win_panel = curses.panel.new_panel(err_win)
                    err_win.box('!', '!')
                    err_win.addstr(max_y // 2, max_x // 2 - 10,
                                   'WINDOW TOO SMALL')
                    err_win.noutrefresh()
                continue

            # resize each window
            try:
                memory_win.resize(
                        lower_win_height,
                        max_x,
                        max_y - lower_win_height,
                        0)
                text_win.resize(
                        max_y - lower_win_height,
                        max_x - upper_win_width,
                        0,
                        0)
                register_win.resize(
                        max_y - lower_win_height,
                        upper_win_width,
                        0,
                        max_x - upper_win_width)

                refresh = True
                err_win_panel = None
            except Exception as e:
                traceback.print_exception(e, file=sys.stderr)
                if not err_win_panel:
                    err_win = curses.newwin(0, 0)
                    err_win_panel = curses.panel.new_panel(err_win)
                    err_win.addstr(max_y // 2, max_x // 2, 'NCURSES ERROR')
                    err_win.noutrefresh()

        elif inp == ord('p'):
            parse_mode = not parse_mode
            refresh = True

        elif inp == ord('q'):
            return


def main():
    try:
        wrapper(_main)
    except Exception as e:
        print(e)
        raise e


if __name__ == '__main__':
    main()

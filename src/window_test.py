#!/usr/bin/env python3

import unittest

from window import Window


class WindowTest(unittest.TestCase):
    def test_init(self):
        sentinel = list()
        cpu = CPU(sentinel)

        self.assertEqual(STACK, cpu.sp)
        self.assertEqual(1, len(cpu.states))
        self.assertEqual(sentinel, cpu.instructions)
        self.assertEqual(None, cpu.data)

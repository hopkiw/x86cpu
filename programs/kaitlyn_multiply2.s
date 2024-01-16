  mov  ax,0x5556
  push ax
  mov  ax,0x3
  push ax
  mov  ax,0x0
start:
  pop  bx
  pop  ax
  mul  bx
  hlt

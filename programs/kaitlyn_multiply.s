  mov  ax,0x5556
  push ax
  mov  ax,0x3
  push ax
  mov  ax,0x0
start:
  pop  ax
  pop  cx
mult:
  add  bx,cx
  mov  dx,bx
  sub  ax,0x1
  je   done
  mov  bx,dx
  jmp  mult
done:
  mov  ax,dx
  hlt

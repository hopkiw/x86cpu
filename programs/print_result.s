  .data
mystr:
  .string "The result is: 0x"
  .string "second string"
  .text
  push  0x31
  push  0x4
  call  mul
  push  ax
  call  print
  nop
  hlt
mul:
  push  bp
  mov   bp,sp
  mov   ax,[bp+0x4]
  mov   bx,[bp+0x6]
  mul   bx
  pop   bp
  ret
  hlt
print:
  push  bp
  mov   bp,sp
  push  mystr
  call  strlen
  pop   bx
  push  ax  ; len
  push  mystr
  call  _print
  mov   ax,[bp+0x4]
  mov   bx,sp
  push  ax
  push  bx
  call  hex
  pop   bx
  push  2
  push  bx
  call  _print
  pop   bx
  pop   bx
  pop   bx
  pop   bx
  pop   bx
  pop   bp
  ret
  hlt
hex:
  push  bp
  mov   bp,sp
  mov   bx,[bp+0x4]
  mov   ax,[bp+0x6]
  mov   cx,0x10
  mov   dx,0
  div   cx
  cmp   ax,0xa
  jg    hex_addalpha
  add   ax,0x30
  jmp   hex_dx
hex_addalpha:
  add   ax,0x57
hex_dx:
  cmp   dx,0xa
  jg    hex_addalpha2
  add   dx,0x30
  jmp   hex_push
hex_addalpha2:
  add   dx,0x57
hex_push:
  mov   [bx],ax
  mov   [bx+0x1],dx
  pop   bp
  ret
  hlt
_print:
  push  bp
  mov   bp,sp
  mov   bx,[bp+0x4]
  mov   cx,[bp+0x6]
  mov   dx,0
_print_loop:
  mov   ax,[bx+dx]
  out   0x20,al
  add   dx,0x1
  cmp   cx,dx
  jne   _print_loop
  pop   bp
  ret
  hlt
strlen:
  push  bp
  mov   bp,sp
  mov   ax,[bp+0x4]
  mov   cx,0
strlen_cmp:
  mov   bx,[ax+cx]
  and   bx,0xff00
  je    strlen_done
  add   cx,0x1
  jmp   strlen_cmp
strlen_done:
  mov   ax,cx
  add   ax,1
  pop   bp
  ret 
  hlt

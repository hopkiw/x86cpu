.data
  .string "Hello, world!"
  .string "This is another string"
  .short  0x4342
  .string "Third string here"
.text
  mov  ax,0x6b00
  push ax
  mov  ax,0x0
  ; start of program
  call strlen
  jmp rlydone
strlen:
  ; top of stack:  return address
  ; below that: arguments to function
  mov  bx,[sp+0x2]
loop:
  movb al,[bx]
  add  cx,0x1
  sub  ax,0x0
  je   done
  add  bx,0x1
  jmp  loop
done:
  mov  ax,cx
  ret
rlydone:
  hlt

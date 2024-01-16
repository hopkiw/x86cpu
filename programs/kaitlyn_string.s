.data
  .string "Hello, world!"
  .string "This is another string"
somelabel:
  .short  0x4342
string3:
  .string "Third string here"
.text
  mov  ax,somelabel
  push ax
  mov  ax,0x0
  ; start of program
  call printstr
  jmp  rlydone
printstr:
  mov  bx,[sp+0x2]
loop:
  movb al,[bx]
  sub  ax,0x0
  je   done
  out  0x20,al
  mov  ax,0x0
  add  bx,0x1
  jmp  loop
done:
  ret
rlydone:
  hlt

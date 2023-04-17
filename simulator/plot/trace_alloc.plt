clear
reset

#set term pdfcairo enhanced font "Times-New-Roman, 22" size 4,3
set term png
set output "./trace_alloc.png"

set border 4095
set grid

set ylabel "Allocated GPUs"
set xlabel "Time (s)"

set xlabel offset 0,0.5
set ylabel offset 1,0 

#set ylabel offset 2

#set key outside top horizontal reverse
set key inside top right reverse
#set key Left width 20
set key samplen 2
#set logscale x 10
#set xtics (1,10,100,1000,1e4)
#set xrange [1:100]
set xtics (0,2000,4000,6000)
# sset xrange [0:2000]
set yrange [0:90]
set ytics mirror
# set logscale x 2

# set size ratio 0.5

#set grid ytics lc rgb "#bbbbbb" lw 1 lt 0
#set grid xtics lc rgb "#bbbbbb" lw 1 lt 0

set style line 1 lt 1 lc rgb 'blue'  lw 3   # --- blue
set style line 2 lt 2 lc rgb 'red' lw 3   # --- red
set style line 3 lt 3 lc rgb 'green' lw 3   # --- red
set style line 4 lt 4 lc rgb 'black' lw 3   # --- red

plot 'trace_alloc.csv' u 1:2 title 'EasyScale_{homo}' w line ls 1, \
                   '' u 1:3 title 'EasyScale_{heter}' w line ls 2

exit

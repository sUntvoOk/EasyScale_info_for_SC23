clear
reset

#set term pdfcairo monochrome enhanced font "Times-New-Roman, 22"
set term png
set output "./jct_makespan.png"

set border 4095
#set logscale y

set xtics nomirror
# rotate by 14 right 
set ytics mirror


set ylabel "Time (s)"
#set xlabel "# of servers"
#set ylabel offset 2

#set key outside top horizontal
set key inside top right
#set key reverse outside vertical bottom Left center
set style data histograms
#set style histogram cluster gap 0.1
#set boxwidth 0.8
set style fill pattern 5 border
#set tics scale 0.0
#set yrange [0.5:1.001]
#set yrange [0:1900]
#set ytics (0, 400, 800, 1200, 1600)
set logscale y 10
set format y '%.0e'

set size ratio 0.9
#set size ratio 1

set xtics offset 2
set xtics nomirror rotate by 25 right

set grid ytics lc rgb "#bbbbbb" lw 1 lt 0


plot newhistogram lt 1, \
     'jct_makespan.csv' index 0 u 3 title "Makespan" fs pattern 0 ls -1, \
     'jct_makespan.csv' index 0 u 2:xtic(1) title "JCT" lt 1
     
exit

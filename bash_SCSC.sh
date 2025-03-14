
epoch=300
time=300
stop=both

n=1000
m=1000

mem_y=0

pretrain=0

# GDA
opt_x=GD
opt_y=GD
lr_x=0.5
lr_y=0.5
num_y=1

# GDN
opt_x=GD
opt_y=Newton
lr_x=0.5
lr_y=1
num_y=1

# CN
# opt_x=Newton
# opt_y=Newton
# lr_x=1
# lr_y=1
# num_y=1

# GDBFGS
# opt_x=GD
# opt_y=BFGS
# lr_x=1
# lr_y=0.5
# num_y=3

# CBFGS
# opt_x=BFGS
# opt_y=BFGS
# lr_x=0.5
# lr_y=0.5
# num_y=2

# GDLBFGS
# opt_x=GD
# opt_y=LBFGS
# lr_x=1
# lr_y=0.5
# num_y=3
# mem_y=3

# CLBFGS
# opt_x=LBFGS
# opt_y=LBFGS
# lr_x=0.5
# lr_y=0.5
# num_y=2
# mem_y=20

python SCSC.py --epoch $epoch \
               --time $time\
               --stop $stop\
               --n $n \
               --m $m \
               --opt_x $opt_x \
               --opt_y $opt_y \
               --lr_x $lr_x \
               --lr_y $lr_y \
               --num_y $num_y \
               --mem_y $mem_y \
               --pretrain $pretrain
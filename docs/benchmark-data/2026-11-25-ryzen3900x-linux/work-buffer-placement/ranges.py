import sys
rows=[l.split() for l in open(sys.argv[1]).read().strip().split("\n")[1:]]
thr=float(sys.argv[2]) if len(sys.argv)>2 else 1.10
bad=[(int(a),float(r)) for a,t,r in rows if float(r)>thr]
# merge into intervals
iv=[];
step=int(rows[1][0])-int(rows[0][0])
for a,r in bad:
    if iv and a-iv[-1][1]<=step: iv[-1][1]=a; iv[-1][2]=max(iv[-1][2],r)
    else: iv.append([a,a,r])
print(sys.argv[1], "best", open(sys.argv[1]).readline().strip().split("best")[1], " slow ranges (>%.2fx): "%thr, len(iv), "ranges, %d of %d offsets"%(len(bad),len(rows)))
for a,b,r in iv: print(f"   dy {a}..{b}  worst {r:.2f}x")

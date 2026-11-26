import csv,glob,statistics as st,collections
def med(pattern):
    d=collections.defaultdict(list)
    for f in glob.glob(pattern):
        for r in csv.DictReader(open(f)):
            if r['status']=='ok': d[(r['library'],r['kind'],int(r['n']))].append(float(r['median_ns'])/1000)
    return {k:(st.median(v),min(v),max(v)) for k,v in d.items()}
for label,pat in (("1 thread","t1-*.csv"),("16 threads","t16-*.csv")):
    m=med(pat)
    print("==",label,"(median of 3 rounds, us; ratios = time / reference, <1 means faster)")
    print(f"{'kind':8}{'N':>9} {'avx2':>9} {'avx512':>9} {'fftw':>9} {'512/avx2':>9} {'512/fftw':>9} {'avx2/fftw':>9}  spread512")
    rs=[];rf=[];rfa=[]
    for k,n in sorted({(k,n) for (_,k,n) in m}):
        a=m.get(("OTFFT-avx2",k,n)); b=m.get(("OTFFT-avx512",k,n)); f=m.get(("FFTW-measure",k,n))
        if not (a and b): continue
        r1=b[0]/a[0]; rs.append(r1)
        r2=b[0]/f[0] if f else float('nan'); r3=a[0]/f[0] if f else float('nan')
        if f: rf.append(r2); rfa.append(r3)
        print(f"{k:8}{n:>9} {a[0]:9.1f} {b[0]:9.1f} {(f[0] if f else float('nan')):9.1f} {r1:9.2f} {r2:9.2f} {r3:9.2f}  {(b[2]-b[1])/b[0]:.0%}")
    print(f"avx512/avx2: median {st.median(rs):.2f} range {min(rs):.2f}-{max(rs):.2f}; faster (<0.98) in {sum(x<0.98 for x in rs)}, slower (>1.02) in {sum(x>1.02 for x in rs)} of {len(rs)}")
    if rf: print(f"avx512/fftw: median {st.median(rf):.2f}; avx2/fftw: median {st.median(rfa):.2f}")

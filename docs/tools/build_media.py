import json, os, glob, base64, io
import numpy as np
from PIL import Image
from scipy import ndimage

ROOT="/mnt/ceph-AT/private/yansongning/AgenticRL4UGC/ai4ugc_map_gen/VibeWorlding-Gym"
SEED="/mnt/ceph-AT/private/yansongning/AgenticRL4UGC/ai4ugc_map_gen/VibeWorlding/raw_data/seed_3dworld"
OUT=os.path.join(ROOT,"docs/web")

TERR={"平原":"Plain","沙漠":"Desert","雪地":"Snowfield","水边":"Waterside","岛屿":"Island"}
GROUND={"草地":"Grass","石地":"Stone","沙地":"Sand","泥地":"Mud","雪地":"Snow","水面":"Water"}
VIEW=["Front","Back","Left","Right","Top"]

def subject_box(im,tol=14):
    a=np.asarray(im.convert("RGB"),dtype=np.int16); h,w,_=a.shape; k=12
    c=np.concatenate([a[:k,:k].reshape(-1,3),a[:k,-k:].reshape(-1,3),
                      a[-k:,:k].reshape(-1,3),a[-k:,-k:].reshape(-1,3)])
    bg=np.median(c,axis=0); m=(np.abs(a-bg).sum(axis=2))>tol
    ys,xs=np.where(m)
    return (0,0,w,h) if len(ys)==0 else (int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1)

def tile(path,TW,TH,margin=0.10):
    im=Image.open(path).convert("RGB"); W,H=im.size
    x0,y0,x1,y1=subject_box(im); cx,cy=(x0+x1)/2,(y0+y1)/2
    bw,bh=(x1-x0)*(1+2*margin),(y1-y0)*(1+2*margin); ar=TW/TH
    if bw/bh<ar: bw=bh*ar
    else: bh=bw/ar
    bw,bh=min(bw,W),min(bh,H)
    if bw/bh<ar: bw=bh*ar
    else: bh=bw/ar
    l=min(max(cx-bw/2,0),W-bw); t=min(max(cy-bh/2,0),H-bh)
    return im.crop((int(l),int(t),int(l+bw),int(t+bh))).resize((TW,TH),Image.LANCZOS)

def fit(path,TW,TH,margin=0.06):
    """Fit the whole subject inside the tile, padding with the render's own bg colour."""
    im=Image.open(path).convert("RGB"); W,H=im.size
    a=np.asarray(im,dtype=np.int16); k=12
    c=np.concatenate([a[:k,:k].reshape(-1,3),a[:k,-k:].reshape(-1,3),
                      a[-k:,:k].reshape(-1,3),a[-k:,-k:].reshape(-1,3)])
    bg=tuple(int(v) for v in np.median(c,axis=0))
    x0,y0,x1,y1=subject_box(im)
    sub=im.crop((x0,y0,x1,y1))
    avail_w,avail_h=TW*(1-2*margin),TH*(1-2*margin)
    sc=min(avail_w/sub.width, avail_h/sub.height)
    nw,nh=max(1,round(sub.width*sc)),max(1,round(sub.height*sc))
    out=Image.new("RGB",(TW,TH),bg)
    out.paste(sub.resize((nw,nh),Image.LANCZOS),((TW-nw)//2,(TH-nh)//2))
    return out

def fill(path,tol=14):
    im=Image.open(path).convert("RGB"); im.thumbnail((320,180))
    a=np.asarray(im,dtype=np.int16); k=6
    c=np.concatenate([a[:k,:k].reshape(-1,3),a[:k,-k:].reshape(-1,3),
                      a[-k:,:k].reshape(-1,3),a[-k:,-k:].reshape(-1,3)])
    bg=np.median(c,axis=0)
    return float(((np.abs(a-bg).sum(axis=2))>tol).mean())

def actors(sid):
    p=os.path.join(SEED,sid,"pcg_scene.json")
    try: j=json.load(open(p,encoding='utf-8'))
    except Exception: return 0
    a=[]
    if isinstance(j,list):
        for b in j:
            if isinstance(b,dict): a+=b.get("actors",[])
    elif isinstance(j,dict): a=j.get("actors",[])
    return len(a)

def fam(t):
    t=(t or "").lower()
    for k,v in [("changan","Ancient City"),("chang_an","Ancient City"),("prosperous","Ancient City"),
        ("temple","Temple"),("garden","Classical Garden"),("tomb","Tomb & Ruins"),("ruins","Tomb & Ruins"),
        ("pyramid","Tomb & Ruins"),("oasis","Desert Oasis"),("fishing","Waterside"),("venice","Waterside"),
        ("beach","Waterside"),("sea","Waterside"),("onsen","Waterside"),("fuji","Waterside"),
        ("snow","Snow"),("winter","Snow"),("villa","Villa Town"),("subway","Modern City"),
        ("airport","Modern City"),("supermarket","Modern City"),("parking","Modern City"),
        ("modern","Modern City"),("city","Modern City"),("amusement","Amusement Park"),
        ("playground","Playground"),("school","Playground"),("classroom","Playground"),
        ("army","Military"),("military","Military"),("march","Military"),("siege","Military"),
        ("camp","Military"),("battle","Military"),("wasteland","Wasteland"),("western","Wild West"),
        ("wild_west","Wild West"),("train","Wild West"),("primitive","Primitive Tribe"),
        ("tribe","Primitive Tribe"),("graveyard","Graveyard"),("grassland","Countryside"),
        ("summer_field","Countryside"),("harvest","Countryside"),("desert","Desert Oasis"),
        ("ancient","Ancient City")]:
        if k in t: return v
    return "Mixed Scene"

def titleize(theme,f):
    t=(theme or "").replace("_"," ").strip()
    if not t or t in ("default scene","default_scene"): return f
    return " ".join(w.capitalize() for w in t.split())

# ---------- scan ----------
rows=[]
for d in os.listdir(SEED):
    p=os.path.join(SEED,d)
    if not os.path.isdir(p): continue
    q=os.path.join(p,"query.json"); mp=os.path.join(p,"init_map.json")
    ims=sorted(glob.glob(os.path.join(p,"image","*.jpg")))
    if not (os.path.exists(q) and os.path.exists(mp) and len(ims)>=5): continue
    try:
        qj=json.load(open(q,encoding='utf-8')); mj=json.load(open(mp,encoding='utf-8'))
    except Exception: continue
    info=mj.get("地图信息",{}) if isinstance(mj,dict) else {}
    sz=info.get("size",{}) or {}
    fs=[fill(x) for x in ims[:5]]
    f=fam(qj.get("theme"))
    rows.append(dict(id=d, theme=qj.get("theme") or "", fam=f,
        title=titleize(qj.get("theme"),f),
        terr=TERR.get(qj.get("terrain_type"),qj.get("terrain_type") or ""),
        ground=GROUND.get(info.get("ground"),info.get("ground") or ""),
        span=(f"{sz.get('x')}x{sz.get('y')} m" if sz.get('x') else ""),
        n=actors(d), fill=fs, best=int(np.argmax(fs[:4])), maxfill=max(fs[:4])))
print("scanned",len(rows))

# ---------- curate gallery ----------
elig=[r for r in rows if r['maxfill']>0.14 and r['n']>=120]
from collections import defaultdict
byf=defaultdict(list)
for r in elig: byf[r['fam']].append(r)
pick=[]
for k,v in byf.items():
    v.sort(key=lambda r:-(r['n']*r['maxfill'])); pick+=v[:2]
pick.sort(key=lambda r:-(r['n']*r['maxfill']))
print("gallery worlds:",len(pick),"families:",len(byf))

TW,TH=896,504
meta=[]
for r in pick:
    src=os.path.join(SEED,r['id'],"image",f"debug_cmd_data_result_{r['best']}.jpg")
    dst=os.path.join(OUT,"worlds",f"w{r['id']}.jpg")
    tile(src,TW,TH).save(dst,quality=80,optimize=True,progressive=True)
    meta.append({k:r[k] for k in ("id","title","fam","terr","ground","span","n")})
tot=sum(os.path.getsize(os.path.join(OUT,"worlds",f)) for f in os.listdir(os.path.join(OUT,"worlds")))
print("world tiles: %d files, %.1f MB"%(len(meta),tot/1e6))

# ---------- 5-view showcase ----------
show=[r for r in pick if r['maxfill']>0.3][:2]
views=[]
for r in show:
    vs=[]
    for v in range(5):
        src=os.path.join(SEED,r['id'],"image",f"debug_cmd_data_result_{v}.jpg")
        dst=os.path.join(OUT,"views",f"{r['id']}_{v}.jpg")
        fit(src,640,360).save(dst,quality=80,optimize=True,progressive=True)
        vs.append({"v":VIEW[v],"i":v})
    views.append(dict(id=r['id'],title=r['title'],n=r['n'],terr=r['terr'],views=vs))
print("showcase:",[v['id'] for v in views])

stats=dict(worlds_total=len(rows))
with open(os.path.join(OUT,"data.js"),"w",encoding='utf-8') as f:
    f.write("window.VW_WORLDS=%s;\n"%json.dumps(meta,ensure_ascii=False))
    f.write("window.VW_VIEWS=%s;\n"%json.dumps(views,ensure_ascii=False))

print("data.js %.2f MB"%(os.path.getsize(os.path.join(OUT,"data.js"))/1e6))

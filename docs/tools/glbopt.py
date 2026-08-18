"""Repack a GLB: downscale oversized textures and re-encode PNG->JPEG.
Low-poly assets here ship 4096^2 PNG atlases on ~5K-tri meshes, which is
~99% of the file size. Geometry is left untouched."""
import struct, json, io, os

def _pad(b, n=4, fill=b'\x00'):
    r = len(b) % n
    return b + fill*(n-r) if r else b

def read_glb(path):
    d=open(path,'rb').read()
    magic,ver,length=struct.unpack_from("<III",d,0)
    assert magic==0x46546C67, "not a glb"
    off=12; J=None; B=b''
    while off < length:
        clen,ctype=struct.unpack_from("<II",d,off)
        body=d[off+8:off+8+clen]
        t=ctype.to_bytes(4,'little')
        if t==b'JSON': J=json.loads(body.decode('utf-8'))
        elif t[:3]==b'BIN': B=body
        off += 8+clen+((4-clen%4)%4 if clen%4 else 0)
    return J,B

def write_glb(path, J, B):
    j=_pad(json.dumps(J,separators=(',',':')).encode('utf-8'), 4, b' ')
    b=_pad(B, 4)
    total=12+8+len(j)+8+len(b)
    with open(path,'wb') as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(j), 0x4E4F534A)); f.write(j)
        f.write(struct.pack("<II", len(b), 0x004E4942)); f.write(b)
    return total

def optimize(src, dst, max_tex=1024, quality=82):
    from PIL import Image
    J,B = read_glb(src)
    images = J.get("images", [])
    if not images:
        return write_glb(dst,J,B), 0

    # Which images are used as a normal/orm map? Those tolerate JPEG poorly at
    # low quality, but these assets are basecolor-only in practice.
    views = J["bufferViews"]
    # Rebuild the binary chunk: keep every non-image view, replace image views.
    keep=[]           # (view_index, bytes)
    newimg={}         # image_index -> (bytes, mime)
    imgviews={}
    for ii,im in enumerate(images):
        if "bufferView" not in im: continue
        imgviews[im["bufferView"]]=ii

    for vi,v in enumerate(views):
        off=v.get("byteOffset",0); ln=v["byteLength"]
        data=B[off:off+ln]
        if vi in imgviews:
            ii=imgviews[vi]
            try:
                pim=Image.open(io.BytesIO(data))
                w,h=pim.size
                has_alpha = pim.mode in ("RGBA","LA","P") and (
                    pim.mode!="P" or "transparency" in pim.info)
                if has_alpha:
                    pim=pim.convert("RGBA")
                    if max(w,h)>max_tex:
                        sc=max_tex/max(w,h)
                        pim=pim.resize((max(1,round(w*sc)),max(1,round(h*sc))),Image.LANCZOS)
                    buf=io.BytesIO(); pim.save(buf,"PNG",optimize=True)
                    out=buf.getvalue(); mime="image/png"
                else:
                    pim=pim.convert("RGB")
                    if max(w,h)>max_tex:
                        sc=max_tex/max(w,h)
                        pim=pim.resize((max(1,round(w*sc)),max(1,round(h*sc))),Image.LANCZOS)
                    buf=io.BytesIO(); pim.save(buf,"JPEG",quality=quality,optimize=True,progressive=False)
                    out=buf.getvalue(); mime="image/jpeg"
                newimg[ii]=(out,mime); data=out
            except Exception:
                pass
        keep.append((vi,data))

    # reassemble
    nb=bytearray(); newviews=[]
    for vi,data in keep:
        while len(nb)%4: nb.append(0)
        v=dict(views[vi]); v["byteOffset"]=len(nb); v["byteLength"]=len(data)
        v.pop("byteStride",None) if vi in imgviews else None
        newviews.append(v); nb+=data
    J["bufferViews"]=newviews
    for ii,(data,mime) in newimg.items():
        J["images"][ii]["mimeType"]=mime
        J["images"][ii].pop("uri",None)
    J["buffers"]=[{"byteLength":len(nb)}]
    return write_glb(dst,J,bytes(nb)), len(newimg)

if __name__=="__main__":
    import sys
    src,dst=sys.argv[1],sys.argv[2]
    mt=int(sys.argv[3]) if len(sys.argv)>3 else 1024
    q=int(sys.argv[4]) if len(sys.argv)>4 else 82
    a=os.path.getsize(src); n,ni=optimize(src,dst,mt,q)
    print("%.2f MB -> %.2f MB (%.0fx smaller, %d textures)"%(a/1e6,n/1e6,a/max(n,1),ni))

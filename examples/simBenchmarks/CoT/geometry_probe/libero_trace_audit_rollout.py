"""State-timeline, resumable LIBERO V3 rollout collection."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from hashlib import sha256
import json, os, tempfile, time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import numpy as np
from .libero_trace_audit_metrics import align_realized_trace, canonicalize_v3_uvd, compute_anchor_metrics, metrics_to_jsonable

RECORD_VERSION=2
LIBERO_DUMMY_ACTION=np.asarray([0.]*6+[-1.],np.float32)
_CASE_KEYS=("suite","task_id","language","rank_group","initial_state_index","seed","original_success")
_ARRAY_FIELDS=("agent_rgb","wrist_rgb","agent_depth","policy_actions_raw","executed_actions","anchor_steps","predicted_uvd","predicted_uvd_time","predicted_uvd_landmark_ids","predicted_depth_current","predicted_depth_future","realized_uvd","realized_xyz","realized_valid","realized_in_frame","anchor_target_uvd","anchor_target_valid","dense_depth_current_target","dense_depth_future_target","latency_ms","camera_k_agentview_flipped")

@dataclass(frozen=True)
class RolloutRecord:
    agent_rgb:np.ndarray; wrist_rgb:np.ndarray; agent_depth:np.ndarray; policy_actions_raw:np.ndarray; executed_actions:np.ndarray; anchor_steps:np.ndarray; predicted_uvd:np.ndarray; predicted_uvd_time:np.ndarray; predicted_uvd_landmark_ids:np.ndarray; predicted_depth_current:np.ndarray; predicted_depth_future:np.ndarray; realized_uvd:np.ndarray; realized_xyz:np.ndarray; realized_valid:np.ndarray; realized_in_frame:np.ndarray; anchor_target_uvd:np.ndarray; anchor_target_valid:np.ndarray; dense_depth_current_target:np.ndarray; dense_depth_future_target:np.ndarray; latency_ms:np.ndarray; camera_k_agentview_flipped:np.ndarray; metadata:dict[str,object]
    @classmethod
    def array_field_names(cls): return _ARRAY_FIELDS
@dataclass(frozen=True)
class StateTimeline: policy_actions_raw:np.ndarray; executed_actions:np.ndarray; state_count:int

def _libero_action(a):
    a=np.asarray(a,np.float32).reshape(-1)
    if a.shape!=(7,): raise ValueError("policy action must be 7D")
    return np.r_[a[:6],1.-2.*float(a[6]>.5)].astype(np.float32)
def finalize_state_timeline(raw,states):
    raw=np.asarray(raw,np.float32)
    if raw.ndim!=2 or raw.shape[1:]!=(7,) or len(states)!=len(raw)+1: raise ValueError("invalid state timeline")
    return StateTimeline(raw,np.stack([_libero_action(x) for x in raw]),len(states))
def derive_inference_seed(seed,anchor):
    if any(isinstance(x,bool) or not isinstance(x,(int,np.integer)) for x in (seed,anchor)) or seed<0 or not 0<=anchor<65536: raise ValueError("invalid audit seed or anchor")
    return int(seed)<<16|int(anchor)
def build_geometry_request(obs,*,inference_seed,unnorm_key=None):
    images=obs.get("image")
    if not isinstance(images,Sequence) or len(images)!=2: raise ValueError("need primary then wrist image")
    out={"examples":[{"image":[images[0],images[1]],"lang":str(obs.get("lang",""))}],"do_sample":False,"use_ddim":True,"num_ddim_steps":10,"return_geometry":True,"inference_seed":int(inference_seed)}
    if unnorm_key is not None: out["unnorm_key"]=unnorm_key
    return out
def flip_camera_intrinsics(k,*,image_size):
    h,w=image_size;k=np.asarray(k,np.float32)
    if k.shape!=(3,3) or h<2 or w<2: raise ValueError("camera K/image size invalid")
    return np.asarray([[-1,0,w-1],[0,-1,h-1],[0,0,1]],np.float32)@k
def _actions(v,h):
    a=np.asarray(v,np.float32);a=a[0] if a.ndim==3 and a.shape[0]==1 else a
    if a.shape!=(h,7): raise ValueError("action chunk shape")
    return a
def execute_cadenced_actions(*,max_steps,action_horizon,audit_seed,dummy_steps,stabilize,observe,request,execute,unnorm_key=None):
    for _ in range(dummy_steps):stabilize(LIBERO_DUMMY_ACTION.copy())
    anchors=[];raw=[];responses=[];chunk=None
    for step in range(max_steps):
        if step%action_horizon==0:
            anchors.append(step);response=request(build_geometry_request(observe(step),inference_seed=derive_inference_seed(audit_seed,step),unnorm_key=unnorm_key))
            if not isinstance(response,Mapping):raise ValueError("policy response must be mapping")
            data=response.get("data",response)
            if not isinstance(data,Mapping) or "actions" not in data:raise ValueError("policy response lacks actions")
            chunk=_actions(data["actions"],action_horizon);responses.append(data)
        action=chunk[step%action_horizon].copy();raw.append(action)
        if execute(_libero_action(action),step):break
    return np.asarray(anchors,np.int32),np.asarray(raw,np.float32),tuple(responses)

def complete_anchor_targets(*,predicted_uvd,predicted_uvd_time,predicted_uvd_landmark_ids,anchor_steps,realized_uvd,realized_valid,action_horizon,image_size,camera_k=None,depth_dead_zone_m=.002):
    p=np.asarray(predicted_uvd,np.float32);t=np.asarray(predicted_uvd_time);ids=np.asarray(predicted_uvd_landmark_ids);anchors=np.asarray(anchor_steps);n,q=p.shape[:2]
    if p.ndim!=4 or p.shape[2:]!=(3,3) or anchors.dtype!=np.int32 or anchors.shape!=(n,) or t.shape!=(n,q*3) or ids.shape!=t.shape:raise ValueError("invalid V3 anchor arrays")
    targets=np.full_like(p,np.nan);valid=np.zeros(p.shape[:-1],bool);metrics={}
    for i,a in enumerate(anchors):
        c=canonicalize_v3_uvd(p[i].reshape(q*3,3),time_points=q,uvd_time=t[i],uvd_landmark_ids=ids[i]);offs=np.rint(t[i].reshape(q,3)[:,0]*action_horizon).astype(int)
        if np.any(offs<0)|np.any(offs>action_horizon):raise ValueError("V3 time outside horizon")
        target,mask=align_realized_trace(realized_uvd,int(a),offs,step_valid=realized_valid);targets[i],valid[i]=target,mask
        metrics[f"anchor_{i}"]=metrics_to_jsonable(compute_anchor_metrics(c,target,mask,image_size=image_size,camera_k=camera_k,depth_dead_zone_m=depth_dead_zone_m,uvd_time=t[i],uvd_landmark_ids=ids[i]))
    json.dumps(metrics,allow_nan=False);return targets,valid,metrics

def _geometry(g):
    keys={"depth_current","depth_future","uvd","uvd_time","uvd_landmark_ids"}
    if not isinstance(g,Mapping) or set(g)!=keys:raise ValueError("geometry response must have exact Task-1 fields")
    def sq(x):
        x=np.asarray(x)
        if x.ndim<1 or x.shape[0]!=1:raise ValueError("geometry batch")
        return x[0]
    u,t,ids=sq(g["uvd"]),sq(g["uvd_time"]),sq(g["uvd_landmark_ids"]);q=len(u)//3
    canonicalize_v3_uvd(u,time_points=q,uvd_time=t,uvd_landmark_ids=ids)
    def depth(x):
        x=sq(x).astype(np.float32);return x[0] if x.ndim==3 and x.shape[0]==1 else x
    return {"uvd":u.reshape(q,3,3).astype(np.float32),"time":t.astype(np.float32),"ids":ids.astype(np.int64),"current":depth(g["depth_current"]),"future":depth(g["depth_future"])}
def _frame(env,obs,resolution,capture):
    if capture:return capture(env,obs,resolution)
    from robosuite.utils import camera_utils
    from starVLA.gripper_triangle import LANDMARK_BODY_NAMES,project_world_to_agentview_uvd
    rgb=np.ascontiguousarray(np.asarray(obs["agentview_image"],np.uint8)[::-1,::-1]);wrist=np.ascontiguousarray(np.asarray(obs["robot0_eye_in_hand_image"],np.uint8)[::-1,::-1]);depth=np.ascontiguousarray(np.asarray(obs["agentview_depth"],np.float32)[::-1,::-1])
    ids=[env.sim.model.body_name2id(x) for x in LANDMARK_BODY_NAMES];xyz=np.asarray(env.sim.data.body_xpos[ids],np.float32);k=camera_utils.get_camera_intrinsic_matrix(env.sim,"agentview",resolution,resolution);pose=camera_utils.get_camera_extrinsic_matrix(env.sim,"agentview");pix,valid,_=project_world_to_agentview_uvd(xyz[None],k,pose,width=resolution,height=resolution);pix=pix[0];valid=valid[0];pix[:,:2]=resolution-1-pix[:,:2];uvd=pix.copy();uvd[:,:2]/=resolution-1;inf=valid&(uvd[:,:2]>=0).all(1)&(uvd[:,:2]<=1).all(1)
    return {"rgb":rgb,"wrist":wrist,"depth":depth,"xyz":xyz,"uvd":uvd.astype(np.float32),"valid":valid.astype(bool),"in_frame":inf.astype(bool),"k":flip_camera_intrinsics(k,image_size=rgb.shape[:2])}
def _resize_depth(d,shape):
    if d.shape==shape:return d.astype(np.float32)
    from .sim_geometry_utils import resize_depth
    return resize_depth(d,np.isfinite(d)&(d>0),shape)[0].astype(np.float32)

def collect_rollout(case,client,args):
    """Collect S actions and initial+post-action S+1 states; imports simulator lazily."""
    factory=getattr(args,"env_factory",None);capture=getattr(args,"frame_capture",None);res=int(getattr(args,"resolution",256));horizon=int(getattr(args,"action_horizon",8));dummy=int(getattr(args,"dummy_steps",10));limit=int(getattr(args,"max_steps",520))
    if factory:env,obs=factory(case,res)
    else:
        os.environ.setdefault("MUJOCO_GL","egl");from libero.libero import benchmark,get_libero_path;from libero.libero.envs import OffScreenRenderEnv
        suite=benchmark.get_benchmark_dict()[case.suite]();task=suite.get_task(case.task_id);env=OffScreenRenderEnv(bddl_file_name=Path(get_libero_path("bddl_files"))/task.problem_folder/task.bddl_file,camera_names=["agentview","robot0_eye_in_hand"],camera_heights=res,camera_widths=res,camera_depths=True);env.seed(case.seed);env.reset();obs=env.set_init_state(suite.get_task_init_states(case.task_id)[case.initial_state_index])
    try:
        for _ in range(dummy):obs,_,_,_=env.step(LIBERO_DUMMY_ACTION.tolist())
        frames=[_frame(env,obs,res,capture)];raw=[];executed=[];pred=[];lat=[];anchors=[];done=False
        for step in range(limit):
            if step%horizon==0:
                anchors.append(step);f=frames[-1];response=client.predict_action(build_geometry_request({"image":[f["rgb"],f["wrist"]],"lang":case.language},inference_seed=derive_inference_seed(case.seed,step),unnorm_key=getattr(args,"unnorm_key",None)));data=response.get("data",response)
                if not isinstance(data,Mapping) or "actions" not in data or "geometry" not in data:raise ValueError("policy response missing actions/geometry")
                chunk=_actions(data["actions"],horizon);pred.append(_geometry(data["geometry"]));lat.append(0.)
            r=chunk[step%horizon].copy();e=_libero_action(r);raw.append(r);executed.append(e);obs,_,done,_=env.step(e.tolist());frames.append(_frame(env,obs,res,capture))
            if done:break
        anchors=np.asarray(anchors,np.int32);p=np.stack([x["uvd"] for x in pred]);times=np.stack([x["time"] for x in pred]);ids=np.stack([x["ids"] for x in pred]);real=np.stack([x["uvd"] for x in frames]);valid=np.stack([x["valid"] for x in frames]);k=frames[0]["k"];targets,mask,metrics=complete_anchor_targets(predicted_uvd=p,predicted_uvd_time=times,predicted_uvd_landmark_ids=ids,anchor_steps=anchors,realized_uvd=real,realized_valid=valid,action_horizon=horizon,image_size=frames[0]["rgb"].shape[:2],camera_k=k);shape=pred[0]["current"].shape;cur=np.stack([_resize_depth(frames[i]["depth"],shape) for i in anchors]);future=np.full_like(cur,np.nan)
        for i,a in enumerate(anchors):
            if a+horizon<len(frames):future[i]=_resize_depth(frames[a+horizon]["depth"],shape)
        return RolloutRecord(np.stack([x["rgb"] for x in frames]),np.stack([x["wrist"] for x in frames]),np.stack([x["depth"] for x in frames]),np.asarray(raw,np.float32),np.asarray(executed,np.float32),anchors,p,times,ids,np.stack([x["current"] for x in pred]),np.stack([x["future"] for x in pred]),real,np.stack([x["xyz"] for x in frames]),valid,np.stack([x["in_frame"] for x in frames]),targets,mask,cur,future,np.asarray(lat,np.float64),k,{"case":asdict(case),"outcome":{"success":bool(done),"end_reason":"done" if done else "max_steps"},"action_horizon":horizon,"image_size":list(frames[0]["rgb"].shape[:2]),"metrics":metrics,"schema":"state_timeline_v2"})
    finally:env.close()

def _spec(a):return {"shape":list(a.shape),"dtype":a.dtype.str}
def _sha(p):
 d=sha256();
 with p.open("rb") as h:
  for b in iter(lambda:h.read(1<<20),b""):d.update(b)
 return d.hexdigest()
def _strict(x):json.dumps(metrics_to_jsonable(x),allow_nan=False)
def _need(a,shape,dtype,name):
 if a.dtype!=np.dtype(dtype) or a.ndim!=len(shape) or any(x!=y for x,y in zip(a.shape,shape) if y is not None):raise ValueError(f"invalid {name}")
def validate_rollout_record(r):
 v={x:np.asarray(getattr(r,x)) for x in _ARRAY_FIELDS};_need(v["agent_rgb"],(None,None,None,3),np.uint8,"agent_rgb");s,h,w,_=v["agent_rgb"].shape;_need(v["wrist_rgb"],(s,h,w,3),np.uint8,"wrist");_need(v["agent_depth"],(s,h,w),np.float32,"depth");a=s-1
 for x in("policy_actions_raw","executed_actions"):_need(v[x],(a,7),np.float32,x)
 if not np.array_equal(v["executed_actions"],np.stack([_libero_action(x) for x in v["policy_actions_raw"]])):raise ValueError("executed action mismatch")
 for x in("realized_uvd","realized_xyz"):_need(v[x],(s,3,3),np.float32,x)
 for x in("realized_valid","realized_in_frame"):_need(v[x],(s,3),np.bool_,x)
 m=r.metadata
 if not isinstance(m,dict) or set(("case","outcome","action_horizon","image_size","metrics","schema"))-set(m) or m["schema"]!="state_timeline_v2" or set(m["case"])!=set(_CASE_KEYS):raise ValueError("invalid metadata")
 H=m["action_horizon"];anchors=v["anchor_steps"]
 if not isinstance(H,int) or anchors.dtype!=np.int32 or anchors[0]!=0 or np.any(np.diff(anchors)!=H) or np.any(anchors>=a):raise ValueError("invalid anchors")
 n=len(anchors);p=v["predicted_uvd"];q=p.shape[1]
 if p.dtype!=np.float32 or p.shape[0]!=n or p.shape[2:]!=(3,3):raise ValueError("invalid predictions")
 _need(v["predicted_uvd_time"],(n,q*3),np.float32,"times");_need(v["predicted_uvd_landmark_ids"],(n,q*3),np.int64,"ids")
 for x in("anchor_target_uvd",):_need(v[x],p.shape,np.float32,x)
 _need(v["anchor_target_valid"],p.shape[:-1],np.bool_,"target_valid");_need(v["latency_ms"],(n,),np.float64,"latency");_need(v["camera_k_agentview_flipped"],(3,3),np.float32,"K")
 for x in("predicted_depth_current","predicted_depth_future","dense_depth_current_target","dense_depth_future_target"):
  if v[x].dtype!=np.float32 or v[x].shape[0]!=n:raise ValueError("depth schema")
 _strict(m)
def save_rollout_record(path,r):
 validate_rollout_record(r);base=Path(path);base.parent.mkdir(parents=True,exist_ok=True);g=f"{base.name}.v2.{os.urandom(8).hex()}";npz=base.parent/(g+".npz");data=base.parent/(g+".json");manifest=base.with_suffix(".manifest.json");arr={x:np.asarray(getattr(r,x)) for x in _ARRAY_FIELDS};tmp=[]
 try:
  for suffix in(".npz",".json",".manifest.json"):
   h=tempfile.NamedTemporaryFile(dir=base.parent,suffix=suffix,delete=False,mode="wb" if suffix==".npz" else "w",encoding=None if suffix==".npz" else "utf-8");tmp.append(Path(h.name));h.close()
  np.savez_compressed(tmp[0],**arr);payload={"version":2,"npz":npz.name,"data":data.name,"sha256":_sha(tmp[0]),"arrays":{x:_spec(y) for x,y in arr.items()},"metadata":metrics_to_jsonable(r.metadata)};_strict(payload);tmp[1].write_text(json.dumps(payload,allow_nan=False));os.replace(tmp[0],npz);os.replace(tmp[1],data);tmp[2].write_text(json.dumps({"version":2,"generation":g},allow_nan=False));os.replace(tmp[2],manifest)
 except:
  npz.unlink(missing_ok=True);data.unlink(missing_ok=True);raise
 finally:
  for x in tmp:x.unlink(missing_ok=True)
 return npz,manifest
def load_rollout_record(path,*,expected_case=None,expected_config_identity=None):
 base=Path(path);man=base.with_suffix(".manifest.json");pointer=json.loads(man.read_text());
 if pointer.get("version")!=2:raise ValueError("unsupported record")
 g=pointer["generation"];npz=man.parent/(g+".npz");data=man.parent/(g+".json");payload=json.loads(data.read_text())
 if not npz.is_file() or payload.get("version")!=2 or payload.get("npz")!=npz.name or _sha(npz)!=payload.get("sha256"):raise ValueError("partial generation")
 with np.load(npz,allow_pickle=False) as z:arr={x:np.asarray(z[x]) for x in _ARRAY_FIELDS}
 r=RolloutRecord(**arr,metadata=payload["metadata"]);validate_rollout_record(r)
 if expected_case is not None and dict(expected_case)!=r.metadata["case"]:raise ValueError("expected_case mismatch")
 if expected_config_identity is not None and any(r.metadata.get(k)!=v for k,v in expected_config_identity.items()):raise ValueError("expected_config_identity mismatch")
 return r

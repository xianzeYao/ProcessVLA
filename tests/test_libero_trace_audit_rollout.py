from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from examples.simBenchmarks.CoT.geometry_probe.libero_trace_audit_selection import AuditCase
from examples.simBenchmarks.CoT.geometry_probe.libero_trace_audit_rollout import RolloutRecord,collect_rollout,complete_anchor_targets,finalize_state_timeline,flip_camera_intrinsics,load_rollout_record,save_rollout_record,execute_cadenced_actions
def _case():return AuditCase("libero_goal",1,"move","best",0,7,True)
def _record():
 a,h,w=2,3,5;u=np.zeros((1,2,3,3),np.float32);u[...,2]=1;s=np.zeros((3,3,3),np.float32);s[...,2]=1;c=_case()
 return RolloutRecord(np.zeros((3,h,w,3),np.uint8),np.zeros((3,h,w,3),np.uint8),np.ones((3,h,w),np.float32),np.asarray([[0,0,0,0,0,0,0],[0,0,0,0,0,0,1]],np.float32),np.asarray([[0,0,0,0,0,0,1],[0,0,0,0,0,0,-1]],np.float32),np.asarray([0],np.int32),u,np.asarray([[0,0,0,1,1,1]],np.float32),np.asarray([[0,1,2,0,1,2]],np.int64),np.ones((1,h,w),np.float32),np.ones((1,h,w),np.float32),s,np.ones_like(s),np.ones((3,3),bool),np.ones((3,3),bool),s[None,[0,2]],np.ones((1,2,3),bool),np.ones((1,h,w),np.float32),np.ones((1,h,w),np.float32),np.asarray([1.],np.float64),np.asarray([[10,0,2],[0,11,1],[0,0,1]],np.float32),{"case":{"suite":c.suite,"task_id":c.task_id,"language":c.language,"rank_group":c.rank_group,"initial_state_index":c.initial_state_index,"seed":c.seed,"original_success":c.original_success},"outcome":{"success":False,"end_reason":"max_steps"},"action_horizon":2,"image_size":[h,w],"metrics":{},"schema":"state_timeline_v2"})
def test_state_timeline_and_camera_endpoint():
 raw=np.asarray([[0,0,0,0,0,0,0],[0,0,0,0,0,0,1]],np.float32);assert finalize_state_timeline(raw,[0,1,2]).state_count==3
 p=np.zeros((1,2,3,3),np.float32);p[...,2]=1;_,v,m=complete_anchor_targets(predicted_uvd=p,predicted_uvd_time=np.asarray([[0,0,0,1,1,1]],np.float32),predicted_uvd_landmark_ids=np.asarray([[0,1,2,0,1,2]],np.int64),anchor_steps=np.asarray([0],np.int32),realized_uvd=np.ones((3,3,3),np.float32),realized_valid=np.ones((3,3),bool),action_horizon=2,image_size=(3,5),camera_k=np.eye(3,dtype=np.float32));assert v[0,1].all() and np.isfinite(m["anchor_0"]["camera"]["center_mae_mm"])
 np.testing.assert_allclose(flip_camera_intrinsics(np.asarray([[10,0,3],[0,11,2],[0,0,1]],np.float32),image_size=(7,13)),[[-10,0,9],[0,-11,4],[0,0,1]])
def test_generation_manifest_identity_and_interruption(tmp_path,monkeypatch):
 path=tmp_path/"rollout";old=_record();save_rollout_record(path,old);assert load_rollout_record(path,expected_case=old.metadata["case"]).agent_rgb.shape[0]==3
 import examples.simBenchmarks.CoT.geometry_probe.libero_trace_audit_rollout as mod;real=mod.os.replace
 def fail(src,dst):
  if Path(dst)==path.with_suffix(".manifest.json"):raise OSError("interrupted")
  return real(src,dst)
 monkeypatch.setattr(mod.os,"replace",fail)
 with pytest.raises(OSError):save_rollout_record(path,replace(old,latency_ms=np.asarray([2.],np.float64)))
 assert load_rollout_record(path).latency_ms.tolist()==[1.] and len(list(tmp_path.glob("rollout.v2.*")))==2
class _Env:
 def __init__(self):self.received=[];self.index=0;self.closed=False
 def step(self,a):self.received.append(np.asarray(a,np.float32));self.index+=1;return _obs(),0.,False,{}
 def close(self):self.closed=True
def _obs():return {"agentview_image":np.full((3,5,3),11,np.uint8),"robot0_eye_in_hand_image":np.full((3,5,3),22,np.uint8),"agentview_depth":np.ones((3,5),np.float32)}
class _Client:
 def __init__(self):self.requests=[]
 def predict_action(self,r):
  self.requests.append(r);a=np.asarray([[[0,0,0,0,0,0,0],[0,0,0,0,0,0,1]]],np.float32);g={"depth_current":np.ones((1,1,3,5),np.float32),"depth_future":np.ones((1,1,3,5),np.float32),"uvd":np.asarray([[[.5,.5,1]]*6],np.float32),"uvd_time":np.asarray([[0,0,0,1,1,1]],np.float32),"uvd_landmark_ids":np.asarray([[0,1,2,0,1,2]],np.int64)};return {"data":{"actions":a,"geometry":g}} if len(self.requests)%2 else {"actions":a,"geometry":g}
def _capture(env,obs,res):return {"rgb":np.full((3,5,3),11,np.uint8),"wrist":np.full((3,5,3),22,np.uint8),"depth":np.ones((3,5),np.float32),"xyz":np.asarray([[1,0,0],[2,0,0],[3,0,0]],np.float32),"uvd":np.asarray([[.2,.2,1],[.5,.2,1],[.35,.5,1]],np.float32),"valid":np.ones(3,bool),"in_frame":np.ones(3,bool),"k":np.asarray([[-10,0,4],[0,-10,2],[0,0,1]],np.float32)}
def test_collect_rollout_fake_env_state_timeline_and_requests():
 env=_Env();client=_Client();args=SimpleNamespace(env_factory=lambda case,res:(env,_obs()),frame_capture=_capture,resolution=5,action_horizon=2,dummy_steps=10,max_steps=4);r=collect_rollout(_case(),client,args)
 assert r.anchor_steps.tolist()==[0,2] and len(r.executed_actions)==4 and r.agent_rgb.shape[0]==5 and r.realized_uvd.shape[0]==5
 np.testing.assert_array_equal(np.asarray(env.received[10:]),r.executed_actions);np.testing.assert_array_equal(r.executed_actions[:,6],[1,-1,1,-1]);np.testing.assert_array_equal(r.policy_actions_raw[:,6],[0,1,0,1]);assert r.anchor_target_valid[1,1].all() and np.isfinite(r.metadata["metrics"]["anchor_0"]["camera"]["span_mae_mm"])
 assert client.requests[0]["return_geometry"] and client.requests[0]["inference_seed"]==(7<<16) and client.requests[1]["inference_seed"]==(7<<16|2);np.testing.assert_array_equal(client.requests[0]["examples"][0]["image"][0],np.full((3,5,3),11,np.uint8));np.testing.assert_array_equal(client.requests[0]["examples"][0]["image"][1],np.full((3,5,3),22,np.uint8));np.testing.assert_array_equal(r.realized_xyz[0,:,0],[1,2,3]);assert env.closed
def test_execute_cadence_accepts_direct_and_data_wrapped_responses():
 responses=iter([{"actions":np.zeros((2,7),np.float32)},{"data":{"actions":np.zeros((2,7),np.float32)}}]);a,raw,_=execute_cadenced_actions(max_steps=3,action_horizon=2,audit_seed=7,dummy_steps=10,stabilize=lambda _:None,observe=lambda _:{"image":[np.zeros((2,2,3),np.uint8)]*2},request=lambda _:next(responses),execute=lambda *_:False);assert a.tolist()==[0,2] and raw.shape==(3,7)

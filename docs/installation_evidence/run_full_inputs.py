import concurrent.futures,hashlib,json,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]
tasks=['WX-01','WA-01','WA-05','WR-01','WR-08','TS02','TS03','TM01','TM02']
worker=root/'work/check_installed_candidate.py'
cases=[(task,seed) for seed in range(1050,1100) for task in tasks]
started=time.monotonic()
def check(case):
 task,seed=case
 report=root/'reports'/f'{task}-{seed}'/'report.json'
 if report.is_file():
  result=json.loads(report.read_text())
  if result.get('passed'):return result
 cmd=[sys.executable,'-I',str(worker),'--acceptance-root',str(root),'--task',task,'--seed',str(seed),'--development-root','/home/chensihang/RoboCasa-Mem','--original-root','/home/chensihang/membench','--render-device','1']
 with (root/'reports'/f'{task}-{seed}.log').open('w') as stream:
  try:p=subprocess.run(cmd,cwd=root/'work',stdout=stream,stderr=subprocess.STDOUT,timeout=180)
  except subprocess.TimeoutExpired:return {'task_id':task,'seed':seed,'passed':False,'error':'timeout after 180 seconds'}
 return json.loads(report.read_text()) if report.is_file() else {'task_id':task,'seed':seed,'passed':False,'returncode':p.returncode}
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
 for future in concurrent.futures.as_completed([pool.submit(check,case) for case in cases]):
  result=future.result();results.append(result)
  if not result.get('passed') or len(results)%9==0:
   print(json.dumps({'completed':len(results),'passed':sum(bool(r.get('passed')) for r in results),'task':result['task_id'],'seed':result['seed'],'error':result.get('error'),'elapsed_s':round(time.monotonic()-started,1)}),flush=True)
  progress={'completed':len(results),'planned':450,'passed':sum(bool(r.get('passed')) for r in results),'elapsed_s':round(time.monotonic()-started,1),'failures':[r for r in results if not r.get('passed')]}
  (root/'reports/full_input_progress.json').write_text(json.dumps(progress,indent=2)+'\n')
summary={'schema_version':'membench.isolated_installation_coverage/v1','passed':len(results)==450 and all(r.get('passed') for r in results),'planned':450,'completed':len(results),'seed_range':[1050,1100],'tasks':tasks,'action_steps_per_case':3,'worker_sha256':hashlib.sha256(worker.read_bytes()).hexdigest(),'duration_s':time.monotonic()-started,'checks':sorted(results,key=lambda r:(tasks.index(r['task_id']),r['seed']))}
(root/'reports/full_input_coverage.json').write_text(json.dumps(summary,indent=2)+'\n')
print('full_input_passed',summary['passed'],flush=True)
sys.exit(0 if summary['passed'] else 1)

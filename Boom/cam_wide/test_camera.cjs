// Camera contract tests only; no recognition accuracy simulation.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const ts=require('../../webapp/web/node_modules/typescript');
const code=ts.transpileModule(fs.readFileSync('webapp/web/lib/camera.ts','utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS}}).outputText;
async function run(query,caps,settings){
 const calls=[],applied=[];const track={getSettings:()=>settings,getCapabilities:()=>caps,getConstraints:()=>({}),applyConstraints:async c=>applied.push(c)};
 const stream={getVideoTracks:()=>[track]};const scope={exports:{},URLSearchParams,window:{location:{search:query}},navigator:{mediaDevices:{getUserMedia:async c=>{calls.push(c);return stream},getSupportedConstraints:()=>({resizeMode:true})}}};
 vm.runInNewContext(code,scope);await scope.exports.openWidestCamera();return {calls,applied,diag:scope.exports.cameraDiagnostics(stream)};
}
(async()=>{
 let r=await run('?cam=wide',{resizeMode:['none','crop-and-scale'],width:{max:1920},height:{max:1080},zoom:{min:1}},{width:1280,height:720,resizeMode:'none'});
 assert.equal(r.calls.length,1);assert.equal(r.applied.length,0);assert.equal(r.calls[0].video.resizeMode,'none');assert.equal(r.calls[0].video.width.ideal,1280);assert.equal(r.diag.resizeModeNoneSupported,true);assert.equal(r.diag.resizeModeNoneApplied,true);
 r=await run('?cam=wide',{}, {width:720,height:720});assert.equal(r.diag.resizeModeCapabilityPresent,false);assert.equal(r.diag.resizeModeNoneSupported,null);assert.equal(r.diag.resizeModeNoneApplied,null);
 r=await run('?cam=wide',{resizeMode:['crop-and-scale']},{resizeMode:'crop-and-scale'});assert.equal(r.diag.resizeModeNoneSupported,false);assert.equal(r.diag.resizeModeNoneApplied,false);
 r=await run('',{width:{max:1280},height:{max:720}},{frameRate:30});assert.equal(r.calls[0].video.resizeMode,undefined);assert.equal(r.applied[0].width.exact,1280);assert.equal(r.diag,undefined);
 r=await run('?cam=480',{width:{max:1280},height:{max:720}},{frameRate:30});assert.equal(r.applied[0].height.exact,480);
 console.log('PASS: wide one request/no applyConstraints; supported/unsupported/missing diagnostics; legacy default and cam=480');
})().catch(e=>{console.error(e);process.exit(1)});

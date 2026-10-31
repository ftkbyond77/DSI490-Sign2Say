import Foundation
import Vision
let hands=VNDetectHumanHandPoseRequest();hands.maximumHandCount=2;hands.usesCPUOnly=true
let body=VNDetectHumanBodyPoseRequest();body.usesCPUOnly=true
do {
try VNImageRequestHandler(url:URL(fileURLWithPath:CommandLine.arguments[1]),orientation:.up).perform([hands,body])
print("hands",hands.results?.count ?? 0,"body",body.results?.count ?? 0)
} catch { print("Vision failure:",error);exit(1) }

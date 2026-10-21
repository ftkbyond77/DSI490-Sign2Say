import Foundation
import Vision
import AVFoundation
let asset=AVAsset(url:URL(fileURLWithPath:CommandLine.arguments[1]))
let track=asset.tracks(withMediaType:.video)[0]
let reader=try AVAssetReader(asset:asset)
let output=AVAssetReaderTrackOutput(track:track,outputSettings:[kCVPixelBufferPixelFormatTypeKey as String:kCVPixelFormatType_32BGRA])
reader.add(output);reader.startReading()
let sample=output.copyNextSampleBuffer()!
let buffer=CMSampleBufferGetImageBuffer(sample)!
let hands=VNDetectHumanHandPoseRequest();hands.maximumHandCount=2;hands.usesCPUOnly=true
let body=VNDetectHumanBodyPoseRequest();body.usesCPUOnly=true
try VNImageRequestHandler(cvPixelBuffer:buffer,orientation:.up).perform([hands,body])
print("hands",hands.results?.count ?? 0,"body",body.results?.count ?? 0)

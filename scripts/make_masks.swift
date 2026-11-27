// Stage 4b: one silhouette mask per frame, for carving (make_mesh.py --masks).
//
//     swift scripts/make_masks.swift work/obj/dense/0/images work/obj/masks
//
// lib: Apple's Vision framework (macOS 14+). VNGenerateForegroundInstanceMaskRequest
// finds the main subject in a photo, like "lift subject from background" in Photos.
// Run it on the *undistorted* frames (dense/0/images) so the masks line up with
// the pinhole cameras in dense/0/sparse_txt. Writes <frame>.png, white = object.
// If Vision finds several subjects, all of them go in the mask: an extra object
// only means that view carves less, while a missed piece of the object would cut
// a hole. Frames with no subject get no mask and are skipped by the carving.
import CoreImage
import Foundation
import Vision

let args = CommandLine.arguments
guard args.count == 3 else {
    print("usage: swift scripts/make_masks.swift IMAGES_DIR OUT_DIR")
    exit(1)
}
let inDir = URL(fileURLWithPath: args[1])
let outDir = URL(fileURLWithPath: args[2])
try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

let exts: Set<String> = ["jpg", "jpeg", "png"]
let files = try FileManager.default.contentsOfDirectory(atPath: inDir.path)
    .filter { exts.contains(($0 as NSString).pathExtension.lowercased()) }
    .sorted()
let ctx = CIContext()
var missing = 0
var several = 0

for (i, name) in files.enumerated() {
    let request = VNGenerateForegroundInstanceMaskRequest()
    let handler = VNImageRequestHandler(url: inDir.appendingPathComponent(name))
    try handler.perform([request])
    guard let result = request.results?.first, !result.allInstances.isEmpty else {
        missing += 1
        print("  no subject found in \(name), skipped")
        continue
    }
    if result.allInstances.count > 1 { several += 1 }
    let mask = try result.generateScaledMaskForImage(forInstances: result.allInstances,
                                                     from: handler)
    let out = outDir.appendingPathComponent((name as NSString).deletingPathExtension + ".png")
    try ctx.writePNGRepresentation(of: CIImage(cvPixelBuffer: mask), to: out, format: .L8,
                                   colorSpace: CGColorSpaceCreateDeviceGray())
    if (i + 1) % 25 == 0 { print("  \(i + 1)/\(files.count)") }
}
print("Wrote \(files.count - missing) masks to \(outDir.path) "
      + "(\(missing) frames without a subject, \(several) with more than one)")

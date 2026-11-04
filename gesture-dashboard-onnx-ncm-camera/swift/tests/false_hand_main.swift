import Foundation
import simd

struct Fixture: Decodable { let cases: [Case] }
struct Case: Decodable {
    let id: String
    let width, height: Int
    let normalized: [[Float]]
    let world: [[Float]]?
    let faces: [[Float]]
    let reason: String?
}
let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1])))
for row in fixture.cases {
    let p = row.normalized.map { SIMD2<Float>($0[0],$0[1]) }
    let world = row.world?.map { SIMD3<Float>($0[0],$0[1],$0[2]) }
    let faces = row.faces.map { GestureFaceBox(x:$0[0],y:$0[1],width:$0[2],height:$0[3]) }
    let result = GestureFalseHandPolicy.rejection(landmarks:p,world:world,width:row.width,height:row.height,faces:faces)
    precondition(result == row.reason, "Mismatch: \(row.id), got \(String(describing: result))")
}
print("PASS: \(fixture.cases.count) Python/Swift candidate-policy cases")

# VOYA Feature Layout Questions

VietSignAI can now inspect a real VOYA_VSL class file and train a direct
`[T,1605]` source-layout baseline. Canonical conversion to VietSign55
`[T,55,3]` remains blocked until the source layout is confirmed.

Questions for the VOYA_VSL author or an authoritative source:

1. What is the exact order of the 1605 per-frame features?
2. Are the 1605 values exactly `535` landmark triplets in `x,y,z` order?
3. Which landmark blocks are present, and in what order?
4. How many pose landmarks are used: 25, 33, or another count?
5. How many face landmarks are used: 468, 460, or another count?
6. Are left and right hands both 21 landmarks?
7. Are confidence, visibility, or presence values included anywhere?
8. Were the features produced by MediaPipe Holistic, another MediaPipe solution,
   or a custom extractor?
9. Which MediaPipe/version and preprocessing code produced the features?
10. Are coordinates normalized to image size, cropped ROI, signer box, or another
    coordinate system?
11. Are missing landmarks encoded as zeros, repeated previous values, NaN, or
    another sentinel?
12. Is there any signer/video/split metadata associated with each sequence?

Until these answers are available, VietSignAI must not claim that VOYA can be
converted into canonical VietSign55 joint order.

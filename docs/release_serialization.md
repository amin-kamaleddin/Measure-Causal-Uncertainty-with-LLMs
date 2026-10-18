# Release serialization

Only derived CSV/SVG files were normalized for Git: CSV CRLF changed to LF, SVG trailing whitespace removed. All parsed CSV rows were verified identical. Protected scientific outputs, raw responses, sealed predictions, scores, source commitments and frozen algorithms remain untouched. The release manifest records the normalized bytes; original hashes remain in the private execution audit.

Three pre-existing blank EOF lines are deliberately preserved byte-for-byte: benchmark/blind.py and both live/final records.py modules. The strict staged check reports exactly these three legacy style findings and no others. The frozen algorithm is not edited for formatting. A check retaining trailing-space/indentation checks while accepting blank EOF lines passes.

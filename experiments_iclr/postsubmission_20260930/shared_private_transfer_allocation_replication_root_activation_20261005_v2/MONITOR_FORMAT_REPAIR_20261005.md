# Progress observation formatting repair

The 20:15:02 UTC remote observation succeeded, but the local V1 summary failed while decoding a progress file read during its write. The retained `b1_S_end_detached/PROGRESS.json` observation has zero bytes, and the complete transport and live owned process observation remain retained. Repoll at 20:15:11 UTC authenticated the same PID 470117/start ticks 6003585737, matching command/cwd, live state S, and three of ten cells complete. No job was restarted or scored.

V2 changes only local summary decoding: unparseable progress metadata is reported explicitly with retained size/hash. It is not terminal or failure evidence. The remote observation source, SSH route, exact owned process handling, source binding and copied raw bytes remain identical. Sealed V1 is preserved.

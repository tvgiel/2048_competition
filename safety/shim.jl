# shim.jl - runs INSIDE the sandbox. Part of the trusted harness, not of a submission.
#
#     julia --project=/opt/julia_env shim.jl /sub/agent_<name>.jl
#
# Same JSON-lines protocol as shim.py. The agent file is include()d into Main,
# so an `if abspath(PROGRAM_FILE) == @__FILE__` guard in the agent stays false.

using JSON3

const AGENT_PATH = abspath(ARGS[1])

# Private handle for the protocol; then point fd 1 at stderr.
const OUT = let fd = ccall(:dup, Cint, (Cint,), 1)
    Base.fdio(fd, true)
end
ccall(:dup2, Cint, (Cint, Cint), 2, 1)
redirect_stdout(stderr)

send(s::AbstractString) = (write(OUT, s, "\n"); flush(OUT))
fail(msg::AbstractString) = send("{\"error\":" * JSON3.write(last(msg, 800)) * "}")

try
    Base.include(Main, AGENT_PATH)
    isdefined(Main, :train_and_instantiate) || error("submission does not define train_and_instantiate()")
    isdefined(Main, :act) || error("submission does not define act()")
catch e
    fail("import failed: " * sprint(showerror, e, catch_backtrace()))
    exit(2)
end

const AGENT = Ref{Any}(nothing)
try
    AGENT[] = Base.invokelatest(Main.train_and_instantiate)
catch e
    fail("train_and_instantiate failed: " * sprint(showerror, e, catch_backtrace()))
    exit(3)
end

send("{\"ready\":true}")

for line in eachline(stdin)
    msg = JSON3.read(line)
    rows = msg.state
    state = Matrix{Int}(undef, length(rows), length(rows[1]))
    for (i, r) in enumerate(rows), (j, v) in enumerate(r)
        state[i, j] = v
    end
    valid = Int[v for v in msg.valid]
    try
        a = Base.invokelatest(Main.act, AGENT[], state, valid)
        send("{\"action\":" * string(Int(a)) * "}")
    catch e
        fail(sprint(showerror, e, catch_backtrace()))
    end
end

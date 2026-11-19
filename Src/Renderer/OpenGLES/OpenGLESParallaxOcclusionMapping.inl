// POM keeps the existing vertex format and material texture units. Two private
// UBOs supply the checked mesh prefix and height block; loose forward uniforms
// are deliberately irrelevant to this independently authored material shader.
bool ParallaxShaderSupported=false;
GLuint ParallaxProgram=0;
std::array<GLuint,2> ParallaxUniforms{};
void DestroyParallax()
{
    if (ParallaxProgram && GlDeleteProgram) GlDeleteProgram(ParallaxProgram);
    if (GlDeleteBuffers) GlDeleteBuffers(static_cast<GLsizei>(ParallaxUniforms.size()),ParallaxUniforms.data());
    ParallaxProgram=0; ParallaxUniforms={};
}
bool EnsureParallax()
{
    if (!ParallaxProgram) ParallaxProgram=CompileStageProgram("ParallaxOcclusionMapping.glsl");
    if (!ParallaxProgram) return false;
    if (!ParallaxUniforms[0]) GlGenBuffers(static_cast<GLsizei>(ParallaxUniforms.size()),ParallaxUniforms.data());
    return ParallaxUniforms[0] && ParallaxUniforms[1];
}
bool BindParallaxDraw(const RealtimeDrawConstants& draw)
{
    ParallaxConstants height;
    if (!BuildParallaxConstants(BoundPipeline.Parallax,BoundPipeline.ParallaxBoundsMinimum,BoundPipeline.ParallaxBoundsMaximum,height))
        return Fail("POM requires finite nonempty local object bounds");
    GlBindBuffer(GL_UNIFORM_BUFFER,ParallaxUniforms[0]); GlBufferData(GL_UNIFORM_BUFFER,sizeof(draw),&draw,GL_DYNAMIC_DRAW);
    GlBindBufferBase(GL_UNIFORM_BUFFER,0,ParallaxUniforms[0]);
    GlBindBuffer(GL_UNIFORM_BUFFER,ParallaxUniforms[1]); GlBufferData(GL_UNIFORM_BUFFER,sizeof(height),&height,GL_DYNAMIC_DRAW);
    GlBindBufferBase(GL_UNIFORM_BUFFER,1,ParallaxUniforms[1]); return true;
}
